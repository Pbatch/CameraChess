from glob import glob


def main():
    for path in glob('hikaru/crops/*.jpg'):
        print(path)


if __name__ == '__main__':
    main()