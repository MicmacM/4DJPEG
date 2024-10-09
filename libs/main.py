from struct import unpack


marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}

def clamp(color):
    return max(min(color, 255), 0)


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decodeHuffman(self, data):
        offset = 0
        header, = unpack("B",data[offset:offset+1])
        offset += 1

        # Extract the 16 bytes containing length data
        lengths = unpack("BBBBBBBBBBBBBBBB", data[offset:offset+16]) 
        offset += 16

        # Extract the elements after the initial 16 bytes
        elements = []
        for i in lengths:
            elements += (unpack("B"*i, data[offset:offset+i]))
            offset += i 

        print("Header: ",header)
        print("lengths: ", lengths)
        print("Elements: ", len(elements))
        data = data[offset:]
    
    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                len_chunk, = unpack(">H", data[2:4])
                len_chunk += 2
                chunk = data[4:len_chunk]

                if marker == 0xffc4:
                    self.decodeHuffman(chunk)
                data = data[len_chunk:]            
            if len(data)==0:
                break     

if __name__ == "__main__":
    img = JPEG('../assets/panda_test.jpg')
    img.decode()    


class HuffmanTable:
    def __init__(self):
        self.root = []
        self.elements = []
    






# OUTPUT:
# Start of Image
# Application Default Header
# Quantization Table
# Quantization Table
# Start of Frame
# Huffman Table
# Huffman Table
# Huffman Table
# Huffman Table
# Start of Scan
# End of Image