import math
import sys
import cv2
import numpy as np 
import collections

#Huffman Encoding

#Tree-Node Type
class Node:
    def __init__(self,freq):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq
    def isLeft(self):
        return self.father.left == self

def createNodes(freqs):
    return [Node(freq) for freq in freqs]


def createHuffmanTree(nodes):
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item:item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]

def huffmanEncoding(nodes,root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes

# Huffman End
# chars_freqs = [('AA', 10), ('G', 2), ('E', 3), ('K', 3), ('B', 4),
                   # ('F', 4), ('I', 4), ('J', 4), ('D', 5), ('H', 6),
                   # ('N', 6), ('L', 7), ('M', 9), ('C', 2)]
# nodes = createNodes([item[1] for item in chars_freqs])
# root = createHuffmanTree(nodes)
# codes = huffmanEncoding(nodes,root)
# for item in zip(chars_freqs,codes):
    # print 'Character:%s freq:%-2d   encoding: %s' % (item[0][0],item[0][1],item[1])
    
class DCTF3:
    def __init__(self):
        self.constLable = (
            (16, 11, 10, 16, 24, 40, 51, 61),
            (12, 12, 14, 19, 26, 58, 60, 55),
            (14, 13, 16, 24, 40, 57, 69, 56),
            (14, 17, 22, 29, 51, 87, 80, 62),
            (18, 22, 37, 56, 68, 109, 103, 77),
            (24, 35, 55, 64, 81, 104, 113, 92),
            (49, 64, 78, 87, 103, 121, 120, 101),
            (72, 92, 95, 98, 112, 100, 103, 99)
            )
        self.cosmatrix16 = dict()
        for x in xrange(8):
            for y in xrange(8):
                for u in xrange(8):
                    for v in xrange(8):
                        k = (x, y, u, v)
                        ans = math.cos((2 * x + 1) * u * math.pi / 16)
                        ans *= math.cos((2 * y + 1) * v * math.pi / 16)
                        self.cosmatrix16[k] = ans
        def get_next(row, column):
            sum = row + column
            if (row == 0 or row == 7) and column % 2 == 0:
                return row,column+1
            if (column == 0 or column == 7) and row % 2 == 1:
                return row+1,column
            if sum % 2 == 1:
                return row+1,column-1
            return row-1,column+1
        position = []
        row, col = 0, 0
        for i in xrange(64):
            position.append((row, col))
            row, col = get_next(row, col)
        self.position = position
    def trans(self, X):
        dct = self.DCT8(X)
        res = []
        for i in xrange(8):
            row = []
            for j in xrange(8):
                row.append(int(round(dct[i][j] * 1.0 / self.constLable[i][j])))
            res.append(row)
        return res
    #
    def detrans(self, X):
        res = []
        for i in xrange(8):
            row = []
            for j in xrange(8):
                row.append(round(int(round(X[i][j])) * self.constLable[i][j]))
            res.append(row)
        dct = self.IDCT8(res)
        return dct
    def IDCT8(self, S):
        assert len(S) == 8 and len(S[0]) == 8
        res = []
        for x in xrange(8):
            row = []
            for y in xrange(8):
                s = 0.0
                for u in xrange(8):
                    cu = math.sqrt(2) / 2 if u == 0 else 1
                    for v in xrange(8):
                        cv = math.sqrt(2) / 2 if v == 0 else 1
                        s += cu * cv * S[u][v] * self.cosmatrix16[(x, y, u, v)]
                s *= 0.25
                row.append(s)
            res.append(row)
        return res
    def DCT8(self, s):
        assert len(s) == 8 and len(s[0]) == 8
        res = []
        for u in xrange(8):
            cu = math.sqrt(2) / 2 if u == 0 else 1
            row = []
            for v in xrange(8):
                cv = math.sqrt(2) / 2 if v == 0 else 1
                S = 0.0
                for x in xrange(8):
                    for y in xrange(8):
                        S += s[x][y] * self.cosmatrix16[(x, y, u, v)]
                S *= 0.25 * cu * cv
                row.append(S)
            res.append(row)
        return res
    def read(self, filename):
        img = cv2.imread(filename,0)
        m,n = img.shape
        self.m = m 
        self.n = n 
        
        result = []
        row_counts8 = m/8
        column_counts8 = n/8

        for i in range(row_counts8):
            for j in range(column_counts8):        
                temp8x8 = np.zeros((8,8))        
                for k in range(8):
                    for l in range(8):
                        temp8x8[k][l] = img[i*8+k][j*8+l]
                result.append(temp8x8)
        fopen = open("bmp.txt","w")
        for i in result:
            for j in i:
                fopen.write(str(j)+'\n')
            fopen.write('\n')
        print "reading"
        for i in xrange(len(result)):
            result[i] = self.trans(result[i])
        self.mtrx = result
        fopen = open("read.txt","w")
        for i in result:
            for j in i:
                fopen.write(str(j)+'\n')
            fopen.write('\n')
        print "read over"
    def decCoef(self, ind, i, j, y):
        assert i < 8 and j < 8
        if self.mtrx[ind][i][j] == 0:
            return -1
        if y == 0 and (self.mtrx[ind][i][j] == 1 or self.mtrx[ind][i][j] == -1):
            self.mtrx[ind][i][j] = 0
            return -1
        if (self.mtrx[ind][i][j] & 1) == y:
            return 0
        if self.mtrx[ind][i][j] > 0:
            self.mtrx[ind][i][j] -= 1
        else :
            self.mtrx[ind][i][j] += 1
        return 0
    def implant(self, mstring):
        ind = 0
        mxind = 0
        for thi in mstring:
            y = int(thi)
            while True:
                i, j = self.position[ind]
                ret = self.decCoef(mxind, i, j, y)
                ind += 1
                if ret == 0:
                    break
                if ind == 64:
                    ind = 0
                    mxind += 1
        
        fopen = open("implant.txt","w")
        for i in self.mtrx:
            for j in i:
                fopen.write(str(j)+'\n')
            fopen.write('\n')
        print "implant over"
        
        # Serialization
        sto = []
        cnt = 0
        for i in self.mtrx:
            for j in xrange(64):
                x, y = self.position[j]
                if i[x][y] == 0:
                    cnt += 1
                else:
                    sto.append(cnt)
                    sto.append(i[x][y])
                    cnt = 0
        print len(sto)
        cntsto = collections.Counter(sto)
        ind = dict()
        fre = []
        for indx, i in enumerate(cntsto):
            fre.append((i, cntsto[i]))
            ind[i] = indx
        nodes = createNodes([item[1] for item in fre])
        root = createHuffmanTree(nodes)
        codes = huffmanEncoding(nodes,root)
        seans = ''
        for i in sto:
            seans += codes[ind[i]]
        print seans
        print len(seans)
        # 
        writeabc = ""
        cnt = 0
        val = 0
        for j in seans:
            cnt += 1
            val *= 2
            if j == '1':
                val += 1
            if cnt == 4:
                cnt = 0
                if val < 10:
                    writeabc += str(val)
                else:
                    writeabc += chr(ord('a') + val - 10)
                val = 0
        if cnt > 0:
            print "cnt=", cnt
            val <<= 4 - cnt
            if val < 10:
                writeabc += str(val)
            else:
                writeabc += chr(ord('a') + val - 10)
        print writeabc
        lenWriteabc = len(writeabc)
        if lenWriteabc & 1 == 1:
            writeabc += '0'
        
        bins = writeabc.decode("hex")
        binfile = open("out.bin", "wb")
        binfile.write(bins)
        
    def readstring(self, filename = None):
        if type(filename) == type(""):
            self.read(filename)
        outstr = ""
        for mx in self.mtrx:
            for ind in xrange(64):
                i, j = self.position[ind]
                if mx[i][j] == 0:
                    continue
                outstr += str(mx[i][j] & 1)
        return outstr
    def save(self, outname = "reshow.bmp"):
        m = self.m 
        n = self.n 
        result = []
        for i in xrange(len(self.mtrx)):
            result.append(self.detrans(self.mtrx[i]))
        
        fopen = open("result.txt","w")
        for i in result:
            for j in i:
                fopen.write(str(j)+'\n')
            fopen.write('\n')
        print "saving"
        reshow_mat = np.zeros((m,n))
        row_counts8 = m/8
        column_counts8 = n/8
        for i in range(row_counts8*column_counts8):
            for j in range(8):
                for k in range(8):
                    reshow_mat[i/column_counts8*8+j][(i%column_counts8)*8+k] = result[i][j][k]
        reshow_mat = reshow_mat
        
        cv2.imwrite(outname,reshow_mat)

#
aa = "101011010101001010101010000010101010111010100100001001010101010100100000101010101010101010101010011001010100101010101"
bbb = DCTF3()
bbb.read("1.bmp")
bbb.implant(aa)
bbb.save("2.bmp")
print bbb.readstring()
print "hello world"