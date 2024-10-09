data = bytearray([0xD6, 0xD7])

print(data)
pos = 9
b = data[pos >> 3]
s = 7 - (b & 0x7)
print(b)
print(pos >> 3)
print(data)
print(s)
