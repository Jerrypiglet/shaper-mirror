from hashlib import md5


def get_md5_for_file(file_path):
    m = md5()
    a_file = open(file_path, "rb")
    m.update(a_file.read())
    a_file.close()
    return m.hexdigest()


if __name__ == "__main__":
    a_file_path = "/home/rayc/Projects/shaper/data/modelnet40/modelnet40_fewshot/support_data_smp1_cross0.h5"
    a = get_md5_for_file(a_file_path)
    print("MD5 for {}:".format(a_file_path))
    print(a)
