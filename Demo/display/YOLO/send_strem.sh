ABCNEWS='https://www.youtube.com/watch?v=rQSwh3bgs5k'
taskset --cpu-list 5 env CUDA_VISIBLE_DEVICES=1 ipython send_yolo.py localhost 5001 6  "$ABCNEWS"
