import subprocess


def main():
    subprocess.call(['yt-dlp',
                     '-f', 'bv',
                     '-o', 'data/autolabel/%(title)s.%(ext)s',
                     'https://www.youtube.com/playlist?list=UUvM8shKfqDGpepxKPyhOy_Q',
                     '--restrict-filenames'])


if __name__ == '__main__':
    main()
