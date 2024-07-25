

def main():
    import pyscreenshot

    image = pyscreenshot.grab(bbox=(10, 10, 500, 500))

    # To view the screenshot
    image.show()

    # To save the screenshot
    image.save("screenshot.png")

if __name__ == "__main__":
    main()