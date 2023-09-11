# Create a video from a directory of images
# read from viz_root_dir / dirname and write to a video viz_root_dir / dirname.mp4
    for dirname in Path(viz_root_dir).iterdir():
        if not dirname.is_dir():
            continue
        dirname = dirname.stem
        print(f"processing {dirname}...")
        filename = osp.join(viz_root_dir, f'{dirname}.mp4')
        print(f"writing to {filename}...")
        optical_list = sorted(glob(osp.join(viz_root_dir, dirname, '*.png')))
        img = cv2.imread(optical_list[0])
        height, width, layers = img.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(filename, fourcc, 30, (width, height))
        for image in optical_list:
            video.write(cv2.imread(image))
        cv2.destroyAllWindows()
        video.release()