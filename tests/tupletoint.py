NUM_BUCKETS = (1, 1, 6, 3)



def tuple2int(tuple):
    int_str = ''
    for i in range(len(tuple)):
        int_str += str(tuple[i])
    return int(int_str)

tuple = (0, 0, 2, 2)
integer = int(str(tuple[0])+str(tuple[1])+str(tuple[2])+str(tuple[3]))
print integer

print(tuple2int(NUM_BUCKETS))