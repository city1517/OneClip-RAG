import torch


def cal_peak_score(sim_scores):
    n = sim_scores.shape[0]
    peak_scores = torch.zeros(sim_scores.size(), dtype=sim_scores.dtype, device=sim_scores.device)
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
        peak_scores[i] = 2*sim_scores[i]-llow-rlow
    return peak_scores


def video_chunking(probs, k=None):
    peak_scores = cal_peak_score(probs)
    centers = torch.topk(peak_scores, k).indices.sort()[0]

    boundaries = []
    for i in range(k-1):
        left, right = centers[i], centers[i+1]
        n = right-left
        if n == 1:
            boundaries.append(left.item())
            continue

        left_sum_scores = torch.zeros(n, dtype=probs.dtype, device=probs.device)
        pre = 0
        left_idx = 0
        for j in range(left, right):
            cur_score = probs[j]-probs[j+1]
            left_sum_scores[left_idx] = cur_score + pre
            left_idx += 1
            pre += cur_score
        
        right_sum_scores = torch.zeros(n, dtype=probs.dtype, device=probs.device)
        pre = 0
        right_idx = n-2
        for j in range(right, left+1, -1):
            cur_score = probs[j]-probs[j-1]
            right_sum_scores[right_idx] = cur_score + pre
            right_idx -= 1
            pre += cur_score
        
        combine_scores = torch.zeros(n, dtype=probs.dtype, device=probs.device)
        for j in range(n):
            cur_left = left_sum_scores[j]
            if j == n-1:
                cur_right = 0
            else:
                cur_right = right_sum_scores[j]
            cur_score = cur_left + cur_right
            combine_scores[j] = cur_score
        optimal_bd = combine_scores.argmax()

        boundaries.append(optimal_bd.item()+left.item()+1)
    
    if type(boundaries) == int or boundaries == [] or boundaries[-1] != len(probs)-1:
        boundaries.append(len(probs)-1)

    return boundaries

