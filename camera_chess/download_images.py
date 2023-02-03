import subprocess


def main():
    subprocess.call(['yt-dlp',
                     '-f', 'bv',
                     '-o', 'data/videos/%(title)s.%(ext)s',
                     '-a', 'data/blitz_urls.txt',
                     '--restrict-filenames'])


if __name__ == '__main__':
    main()
