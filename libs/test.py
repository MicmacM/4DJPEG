data = bytearray([0xD6])

print(data)
pos = 0
b = data[pos >> 3]
print(b)
print(data)
print(pos)