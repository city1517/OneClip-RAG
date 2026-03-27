from glob import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--output_dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--eval_type", default="multi_choice", help="The path to save annotation final combined json file.")
    parser.add_argument("--num-chunks", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()


    pred_type=args.eval_type
    ori_dir = args.output_dir
    files = glob(ori_dir + f'/{args.num_chunks}_*')
    if len(files) == 0:
        files = glob(ori_dir + f'/*')
    print(files)
    
    if pred_type == 'multi_choice':
        total_cnt, total_acc = 0, 0
        for file in files:
            lines = open(file, 'r').readlines()
            for line in lines:
                line = eval(line)
                total_cnt += 1

                if line['acc'] == 'True':
                    total_acc += 1
        print(f'total acc: {total_acc / total_cnt}, total_cnt: {total_cnt}')
        

if __name__ == "__main__":
    main()