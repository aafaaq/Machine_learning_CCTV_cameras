def get_name():
    name = input("Enter your admin name: ")
    return name


def get_pass():
    passw = input("Enter password: ")
    return passw


def get_ip():
    print("Note : Enter last part of ip as 192.168.10.**")
    ip = input("Enter ip address: ")
    return ip


def camera_num():
    cam_num = input("Enter camera number: ")
    return cam_num


def list_maker():
    info = []
    info.append(get_name())
    info.append(get_pass())
    info.append(get_ip())
    info.append(camera_num())
    return info

