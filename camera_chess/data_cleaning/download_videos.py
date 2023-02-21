import subprocess


def main():
    subprocess.call(['yt-dlp',
                     '-f', 'bv',
                     '-o', 'data/youtube/%(title)s.%(ext)s',
                     '-a', 'data/video_urls.txt',
                     '--restrict-filenames'])


if __name__ == '__main__':
    main()
