import os
import numpy as np
from .cal_iou import DetectionIoUEvaluator

def load_label_infor(label_file_path, do_ignore=False):
    files = os.listdir(label_file_path)
    img_name_label_dict = {}
    for file in files:
        bbox_infor = []
        with open(os.path.join(label_file_path,file), "r",encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                txt_dict = {}
                substr = line.strip("\n").split(",")
                coord = list(map(int,substr[:8]))
                text = substr[-1]
                ignore = False
                if text == "###" and do_ignore:
                    ignore = True
                txt_dict['ignore'] = ignore
                txt_dict['points'] = np.array(coord).reshape(4,2).tolist()
                txt_dict['text'] = ignore
                bbox_infor.append(txt_dict)
        if do_ignore:
            img_name_label_dict[file.replace('gt_','').replace('.txt','')] = bbox_infor
        else:
            img_name_label_dict[file.replace('res_','').replace('.txt','')] = bbox_infor
    return img_name_label_dict


def cal_det_metrics(gt_label_path, save_res_path):
    """
    calculate the detection metrics
    Args:
        gt_label_path(string): The groundtruth detection label file path
        save_res_path(string): The saved predicted detection label path
    return:
        claculated metrics including Hmean, precision and recall
    """
    evaluator = DetectionIoUEvaluator()
    gt_label_infor = load_label_infor(gt_label_path, do_ignore=True)
    dt_label_infor = load_label_infor(save_res_path)
    results = []
    for img_name in gt_label_infor:
        gt_label = gt_label_infor[img_name]
        if img_name not in dt_label_infor:
            dt_label = []
        else:
            dt_label = dt_label_infor[img_name]
        result = evaluator.evaluate_image(gt_label, dt_label)
        results.append(result)
    methodMetrics = evaluator.combine_results(results)
    return methodMetrics
