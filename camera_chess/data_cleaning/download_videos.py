import subprocess


def main():
    subprocess.call(['yt-dlp',
                     '-f', 'bv',
                     '-o', 'nyh/videos/%(title)s.%(ext)s',
                     'https://www.youtube.com/playlist?list=UUvM8shKfqDGpepxKPyhOy_Q',
                     '--restrict-filenames',
                     '--match-filter', "duration<600"])


if __name__ == '__main__':
    main()
