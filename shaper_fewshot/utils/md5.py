from hashlib import md5


def get_file_md5(fname):
    hash_md5 = md5()
    with open(fname, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


if __name__ == "__main__":
    md5 = get_file_md5(__file__)
    print("md5 of {} is {}".format(__file__, md5))
