import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math

import random
import numpy as np
from glob import glob
from decord import VideoReader, cpu
import sys
import warnings
warnings.filterwarnings("ignore")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_video(vr, args, start=None, end=None, sample_frames=None, is_topk=False, topk_idxs=None):
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    video_time = total_frame_num / fps
    if is_topk == False:
        uniform_sampled_frames = np.linspace(start, min(end, total_frame_num-1), sample_frames, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    else:
        frame_idx = topk_idxs
    frame_time = [i/fps for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        # pred_index = random.choice(all_choices)
        return ''
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def cal_depth_score(sim_scores):
    # n = sim_scores.shape[0]
    # depth_scores = torch.zeros(sim_scores.size(), dtype=sim_scores.dtype, device=sim_scores.device)
    n = len(sim_scores)
    depth_scores = [0]*n
    # clip = min(max(n//10, 2), 5) # adopt clip to improve efficiency
    for i in range(n):
        llow = sim_scores[i]
        for li in range(i-1, -1, -1):
            if sim_scores[li] <= llow:
                llow = sim_scores[li]
            else:
                break
        rlow = sim_scores[i]
        for ri in range(i+1, n):
            if sim_scores[ri] <= rlow:
                rlow = sim_scores[ri]
            else:
                break
        # depth_scores[i] = llow + rlow - 2 * sim_scores[i]
        depth_scores[i] = 2*sim_scores[i]-llow-rlow
    return depth_scores


def segment(probs, alpha=0.5, k=None):
    # input shape: t, d
    # sim_scores = torch.cosine_similarity(features[:-1, :], features[1:, :])
    # depth_scores = cal_depth_score(sim_scores)
    depth_scores = cal_depth_score(probs)

    # centers = torch.topk(depth_scores, k).indices.sort()[0]
    depth_scores = np.array(depth_scores)
    top_k_indices = np.argsort(depth_scores)[-k:]  # 获取最大的 K 个值的索引
    # 排序索引对应的值
    centers = np.sort(top_k_indices)

    

    boundaries = []
    for i in range(k-1):
        left, right = centers[i], centers[i+1]
        n = right-left
        if n == 1:
            boundaries.append(left.item())
            continue

        left_sum_scores = []
        pre = 0
        for j in range(left, right):
            cur_score = probs[j]-probs[j+1]
            left_sum_scores.append(cur_score+pre)
            pre += cur_score
        
        right_sum_scores = [0]
        pre = 0
        for j in range(right, left+1, -1):
            cur_score = probs[j]-probs[j-1]
            right_sum_scores.insert(0, cur_score+pre)
            pre += cur_score
        
        combine_scores = []
        for j in range(n):
            cur_left = left_sum_scores[j]
            if j == n-1:
                cur_right = 0
            else:
                cur_right = right_sum_scores[j]
            cur_score = cur_left + cur_right
            combine_scores.append(cur_score)
        optimal_bd = combine_scores.index(max(combine_scores))

        boundaries.append(optimal_bd+left.item()+1)

    
    # boundaries = boundaries.tolist()
    
    if type(boundaries) == int or boundaries == [] or boundaries[-1] != len(probs)-1:
        boundaries.append(len(probs)-1)

    return boundaries

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    overwrite_config = {}
    overwrite_config["mm_spatial_pool_stride"] = 2
    overwrite_config["mm_spatial_pool_mode"] = "average"
    overwrite_config["mm_pooling_position"] = "before"
    overwrite_config["mm_newline_position"] = "grid"
    overwrite_config["delay_load"] = False
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, overwrite_config=overwrite_config)

    idx_file = args.anno_path
    print("="*30)
    print('load: ', idx_file)
    print("="*30)
    qid2probs = json.load(open(idx_file, 'r')) 
    final_sample_frames = 64

    # Data
    data = json.load(open(args.gt_file, 'r'))
    gt_questions = []
    for item in data:
        question = item['question']
        option = [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(item["candidates"])]
        qid = item['qid']
        video_id = f"{item['category']}/{item['video']}"
        answer = item['answer']
        answer_id = chr(ord("A")+item["candidates"].index(answer))

        duration = item['duration']

        index2ans = {}
        for i in range(len(item["candidates"])):
            idx = chr(ord("A")+i)
            ans = item["candidates"][i]
            index2ans[idx] = ans


        gt_questions.append({
            'qid': qid, 
            'question': question, 
            'option': option, 
            'video_id': video_id, 
            'answer_id': answer_id, 
            'answer': answer, 
            'index2ans': index2ans,
            'duration': duration,
        })
        
    questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        qid = line["qid"]
        answer = line["answer"]
        video_id = line["video_id"]
        answer_id = line["answer_id"]
        option = line["option"]
        index2ans = line["index2ans"]
        duration = line['duration']

        question = [line['question']] + option
        question = '\n'.join(question)
        question = f'{question}\nPlease answer directly with only the letter of the correct option and nothing else.'

        sample_set = {
            "id": qid, 
            "video_id": video_id,
            "question": question, 
            "answer": answer, 
            "answer_id": answer_id, 
            'duration': duration,
        }

        video_path = os.path.join(args.video_dir, video_id)

        # Check if the video exists
        if not os.path.exists(video_path):
            print(f'Miss video {video_path}')
            continue

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = vr.get_avg_fps()
        video_time = total_frame_num / fps
        total_segs = math.ceil(video_time / args.for_get_frames_num) 
        probs = eval(qid2probs[qid]['probs'])
        frame_idx = eval(qid2probs[qid]['frame_idx'])
        boundaries = segment(probs, k=total_segs)

        if args.temporal_type == 'oneclip':
            idx = 0
            oneclip_seg_probs, oneclip_seg_bds = [], []
            for bi in boundaries:
                cur_prob = max(probs[idx:bi+1])

                oneclip_seg_probs.append(cur_prob)
                oneclip_seg_bds.append([idx, bi+1])
                
                idx = bi + 1

            _, oneclip_seg_topk_idxs = torch.topk(torch.tensor(oneclip_seg_probs), k=min(args.topk, len(oneclip_seg_probs)))
            topk_oneclip_max_segs = []
            for idx in oneclip_seg_topk_idxs:
                start_idx, end_idx = oneclip_seg_bds[idx]
                end_idx = min(len(frame_idx)-1, end_idx)
                start_sec, end_sec = frame_idx[start_idx], frame_idx[end_idx]
                topk_oneclip_max_segs.append([start_sec, end_sec])
            seg_duration = sorted(topk_oneclip_max_segs)


            merge_seg_duration = []
            s, e = seg_duration[0]
            cur_seg_num, total_seg_num = 1, len(seg_duration)
            for i in range(1, len(seg_duration)):
                cs, ce = seg_duration[i]
                if e == cs:
                    e = ce
                    cur_seg_num += 1
                else:
                    merge_seg_duration.append([s,e,cur_seg_num])
                    s, e, cur_seg_num = cs, ce, 1
            merge_seg_duration.append([s,e, cur_seg_num])
            seg_duration = merge_seg_duration

            video = []
            for start, end, sn in seg_duration:
                cur_video = load_video(video_path, start=start, end=end, sample_frames=int(final_sample_frames * sn / total_seg_num))
                video.append(cur_video)
            video = np.concatenate(video)
            print(f'seg_duration: {seg_duration}, {len(video)}')


        video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
        video = [video]

        qs = question
        time_instruciton = ''
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + time_instruciton + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + time_instruciton + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        if args.generate_method == 'generate_until':

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=video,

                    modalities= ["video"], 
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

        
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            parsed_pred = parse_multi_choice_response(outputs, ["A", "B", "C", "D", "E"], index2ans)
            sample_set['acc'] = str(parsed_pred == answer_id)   
            sample_set["pred"] = outputs


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()


    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--gt_file", type=str, default="tables/question.jsonl")
    parser.add_argument("--output_dir", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--temporal_type", type=str, default='flat')
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--generate_method", type=str, default='generate_until')
    parser.add_argument("--for_get_frames_num", type=int, default=99)

    parser.add_argument("--anno_path", type=str, default="siglip_full.json")
    args = parser.parse_args()

    eval_model(args)