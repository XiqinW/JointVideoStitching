import run
import video_to_img



def main():
    video_path_a = './video/video_a/'
    save_pat_a = video_path_a
    video_to_img.save_img('./video/video_a/left.mp4', save_pat_a)
    video_path_b = './video/video_b/'
    save_pat_b = video_path_b
    video_to_img.save_img('./video/video_b/right.mp4', save_pat_b)
    run.duanmu(video_path_a, video_path_b, './dmoutput/')


if __name__ == '__main__':
    main()