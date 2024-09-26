from phue import Bridge

def main():
    bridge = Bridge("198.168.1.107")
    bridge.connect()
    lights = bridge.lights
    for light in lights:
        print(f"Light {light.name} is {'on' if light.on else 'off'}")

    for light in lights:
        light.on = False

    for light in lights:
        light.on = True


if __name__ == "__main__":
    main()
