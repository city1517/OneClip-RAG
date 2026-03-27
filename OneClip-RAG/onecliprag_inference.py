import argparse
import torch
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX

import math
from transformers import CLIPProcessor, CLIPModel 
from collections import OrderedDict
import numpy as np
from decord import VideoReader, cpu
import warnings
warnings.filterwarnings("ignore")
import re
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
EN_STOPWORDS = set(stopwords.words('english'))

from tools.chunk_algorithm import video_chunking


def load_video(video_path, start=None, end=None, sample_frames=None):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)

    uniform_sampled_frames = np.linspace(start, min(end, total_frame_num-1), sample_frames, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()

    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def extract_keywords(text: str, max_len: int) -> str:
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in EN_STOPWORDS]
    
    final_text = ' '.join(words)
    
    if len(final_text) > max_len:
        final_text = final_text[:max_len]
    
    return final_text

def split_long_sentence(s, limit):
    s = s.strip()
    if len(s) <= limit:
        return [s]

    comma_parts = [p.strip() for p in s.split(",") if p.strip()]
    chunks = []
    buffer = ""

    for part in comma_parts:
        if len(buffer) + len(part) + 1 <= limit:
            buffer = (buffer + ", " + part).strip(", ").strip()
        else:
            if buffer:
                chunks.append(buffer)
            if len(part) > limit:
                sub_parts = []
                text = part.strip()
                while len(text) > limit:
                    split_point = text.rfind(" ", 0, limit)
                    if split_point == -1:
                        split_point = limit
                    sub_parts.append(text[:split_point].strip())
                    text = text[split_point:].strip()
                if text:
                    sub_parts.append(text)
                chunks.extend(sub_parts)
                buffer = ""
            else:
                buffer = part.strip()
    if buffer:
        chunks.append(buffer)

    final_chunks = [c[:limit].strip() for c in chunks if c.strip()]
    return final_chunks

def instruction_preprocess(instruction: str) -> str:
    instruction = instruction.strip()
    sentences = sent_tokenize(instruction)
    merged_sentences = []
    cur_sent = ""
    max_len = 77
    for sent in sentences:
        processed_text = extract_keywords(sent, 999999)
        short_segments = split_long_sentence(processed_text, max_len)
        for seg in short_segments:
            cur_sent = seg.strip()
            merged_sentences.append(cur_sent)

    merged_sentences = [m[-max_len:].strip() for m in merged_sentences]
    return merged_sentences

def eval_model(args):
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()

    if args.ckpt_path:
        print('=> loading ckpt...')
        latest_checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        model_state_dict = OrderedDict()
        for k, v in latest_checkpoint['state_dict'].items():
            new_key = k.replace('module.', '')
            model_state_dict[new_key] = v
        clip_model.load_state_dict(model_state_dict)

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

    instruction = args.instruction
    video_path = args.video_path
    sample_fps = args.sample_fps
    sec_per_seg = args.sec_per_seg
    
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    duration = total_frame_num / fps
    max_length = math.ceil(duration * sample_fps) 
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_length, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    video = vr.get_batch(frame_idx).asnumpy()

    texts = instruction_preprocess(instruction)

    all_probs = []
    with torch.no_grad():
        for sent in texts:
            inputs = clip_processor(text=[sent], images=video, return_tensors="pt", padding=True)
            for k, v in inputs.items():
                inputs[k] = v.cuda()
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_text.softmax(dim=-1)[0]
            all_probs.append(probs)

    if len(all_probs) > 1:
        probs = torch.stack(all_probs, dim=0).mean(dim=0)
    else:
        probs = all_probs[0]

    total_segs = math.ceil(duration / sec_per_seg) 

    boundaries = video_chunking(probs, k=total_segs)


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
        end_idx = min(max_length-1, end_idx)
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
        cur_video = load_video(video_path, start=start, end=end, sample_frames=int(args.input_frames * sn / total_seg_num))
        video.append(cur_video)
    video = np.concatenate(video)

    video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
    video = [video]

    qs = instruction

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

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video,
            modalities= ["video"], 
            do_sample=False,
            temperature=args.temperature,
            top_p=args.top_p,
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--video_path", type=str, help="Path to your input video file")
    parser.add_argument("--instruction", type=str, help="Your instruction for the video")
    parser.add_argument("--ckpt-path", type=str, default='/path/to/ckpt')
    parser.add_argument("--sec_per_seg", type=float, default=10)
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--input_frames", type=int, default=64)
    parser.add_argument("--topk", type=int, default=16)
    
    args = parser.parse_args()

    eval_model(args)