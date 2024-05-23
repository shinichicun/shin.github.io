---
title: CISCN2024-Crypto
published: 2024-05-24
description: '记录一下'
image: ''
tags: [Crypto]
category: ''
draft: false
---

<!-- toc -->

经典差一点。。。难绷

![image-20240519202858861](CISCN2024-Crypto/image-20240519202858861.png)

## OvO

**题目：**

<details>
    <summary>点击展开代码</summary>


```python
from Crypto.Util.number import *
from secret import flag

nbits = 512
p = getPrime(nbits)
q = getPrime(nbits)
n = p * q
phi = (p-1) * (q-1)
while True:
    kk = getPrime(128)
    rr = kk + 2
    e = 65537 + kk * p + rr * ((p+1) * (q+1)) + 1
    if gcd(e, phi) == 1:
        break
m = bytes_to_long(flag)
c = pow(m, e, n)

e = e >> 200 << 200
print(f'n = {n}')
print(f'e = {e}')
print(f'c = {c}')

"""
n = 111922722351752356094117957341697336848130397712588425954225300832977768690114834703654895285440684751636198779555891692340301590396539921700125219784729325979197290342352480495970455903120265334661588516182848933843212275742914269686197484648288073599387074325226321407600351615258973610780463417788580083967
e = 37059679294843322451875129178470872595128216054082068877693632035071251762179299783152435312052608685562859680569924924133175684413544051218945466380415013172416093939670064185752780945383069447693745538721548393982857225386614608359109463927663728739248286686902750649766277564516226052064304547032760477638585302695605907950461140971727150383104
c = 14999622534973796113769052025256345914577762432817016713135991450161695032250733213228587506601968633155119211807176051329626895125610484405486794783282214597165875393081405999090879096563311452831794796859427268724737377560053552626220191435015101496941337770496898383092414492348672126813183368337602023823
"""
```

</details>

题目有个关键信息：
$$
e = 65537 + kk * p + rr * ((p+1) * (q+1)) + 1~~~~~~~~~①\\rr=kk+2
$$
于是我们可以得到这样两个数：$e//n=rr、~kk=rr-2$

而①式可以转换成这样：
$$
e = 65537 + kk * p + rr * ((p+1) * (n/p+1)) + 1
$$
所以我们可以得到这个方程：
$$
e*p = 65538*p + kk * p^2 + rr * ((p+1) * (n+p))
$$
最后我们直接去解这个方程就可以得到素数p，进而得到flag；这里用到了精度域RealField(1000)，所以可以不用考虑small_roots()需要求几位（个人理解，仅供参考）

exp：

<details>
    <summary>点击展开代码</summary>


```python
# sage10.3
from Crypto.Util.number import *


def partial_p(p0, kbits, n):
    PR.<x> = PolynomialRing(Zmod(n))
    nbits = n.nbits()
    f = x + p0
    f = f.monic()

    roots = f.small_roots(X=2^(kbits), beta=0.4)

    if roots:
        x0 = roots[0]
        p = gcd(x0 + p0, n)
        return int(p)

def find_p(eh, kbits, n):
    P.<x> = PolynomialRing(RealField(1000))
    f = (65538+rr*(n+1))*x + (kk+ rr) * x**2 + rr * n - eh*x
    res = f.monic().roots()
    if res:
        for i in res:
            ph = int(i[0])
            print(ph)
            p = partial_p(ph, kbits, n)
            if p and isPrime(p):
                return p

n = 111922722351752356094117957341697336848130397712588425954225300832977768690114834703654895285440684751636198779555891692340301590396539921700125219784729325979197290342352480495970455903120265334661588516182848933843212275742914269686197484648288073599387074325226321407600351615258973610780463417788580083967
e = 37059679294843322451875129178470872595128216054082068877693632035071251762179299783152435312052608685562859680569924924133175684413544051218945466380415013172416093939670064185752780945383069447693745538721548393982857225386614608359109463927663728739248286686902750649766277564516226052064304547032760477638585302695605907950461140971727150383104
c = 14999622534973796113769052025256345914577762432817016713135991450161695032250733213228587506601968633155119211807176051329626895125610484405486794783282214597165875393081405999090879096563311452831794796859427268724737377560053552626220191435015101496941337770496898383092414492348672126813183368337602023823
rr = e // n
kk = rr - 2
p = find_p(e, 200, n)
if p:
    q = n // p
    e = 65537 + kk * p + rr * ((p+1) * (q+1)) + 1
    d = inverse(e, (p-1)*(q-1))
    print(long_to_bytes(pow(c, d, n)))
# flag{b5f771c6-18df-49a9-9d6d-ee7804f5416c}
```

</details>

---

## ezrsa（复现）

**题目：**

<details>
    <summary>点击展开代码</summary>


```python
from Crypto.Util.number import *
from Crypto.PublicKey import RSA
import random
from secret import flag

m = bytes_to_long(flag)
key = RSA.generate(1024)
passphrase = str(random.randint(0,999999)).zfill(6).encode()
output = key.export_key(passphrase=passphrase).split(b'\n')
for i in range(7, 15):
    output[i] = b'*' * 64
with open("priv.pem", 'wb') as f:
    for line in output:
        f.write(line + b'\n')
"""
私钥文件
-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3-CBC,435BF84C562FE793

9phAgeyjnJYZ6lgLYflgduBQjdX+V/Ph/fO8QB2ZubhBVOFJMHbwHbtgBaN3eGlh
WiEFEdQWoOFvpip0whr4r7aGOhavWhIfRjiqfQVcKZx4/f02W4pcWVYo9/p3otdD
ig+kofIR9Ky8o9vQk7H1eESNMdq3PPmvd7KTE98ZPqtIIrjbSsJ9XRL+gr5a91gH
****************************************************************
****************************************************************
****************************************************************
****************************************************************
****************************************************************
****************************************************************
****************************************************************
****************************************************************
hQds7ZdA9yv+yKUYv2e4de8RxX356wYq7r8paBHPXisOkGIVEBYNviMSIbgelkSI
jLQka+ZmC2YOgY/DgGJ82JmFG8mmYCcSooGL4ytVUY9dZa1khfhceg==
-----END RSA PRIVATE KEY-----
"""
with open("enc.txt", 'w') as f:
    f.write(str(key._encrypt(m)))
"""
密文 55149764057291700808946379593274733093556529902852874590948688362865310469901900909075397929997623185589518643636792828743516623112272635512151466304164301360740002369759704802706396320622342771513106879732891498365431042081036698760861996177532930798842690295051476263556258192509634233232717503575429327989
"""
```

</details>

题目给了私钥文件和密文，虽然私钥文件不完整，但也不是完全没有有用的信息；然而，这里的私钥文件进行了加密：

```python
passphrase = str(random.randint(0,999999)).zfill(6).encode()
output = key.export_key(passphrase=passphrase).split(b'\n')
```

因为passphrase的范围是[0,999999]，所以我们可以爆破去找passphrase；但我们还是需要看看加密代码。

翻一下export_key的源代码，发现加密的位置在这：

![image-20240519203812924](CISCN2024-Crypto/image-20240519203812924.png)

查看此处函数的源代码：

![image-20240519203919740](CISCN2024-Crypto/image-20240519203919740.png)

对该函数的解释：

> 1，随机生成8bytes的salt
>
> 2，PBKDF1(passphrase, salt, 16, 1, MD5) 得到 key 在末尾并附上PBKDF1(key + passphrase, salt, 8, 1, MD5)（不理解函数的意思也没事，只要知道这里的key是这样的形式就行）
>
> 3，使用3DES进行CBC模式加密（key为第二步的key，iv为salt）：
>
> ​	objenc = DES3.new(key, DES3.MODE_CBC, salt)
>
> ​	data = objenc.encrypt(pad(data, objenc.block_size))

由 `"Proc-Type: 4,ENCRYPTED\nDEK-Info: DES-EDE3-CBC,%s\n\n" %tostr(hexlify(salt).upper())`，可以知道私钥文件里的iv（salt）的值为**435BF84C562FE793**（这里可以看看前面的私钥文件）。

前面也提到passphrase可以进行爆破，于是我们可以通过 ”解密前半来判断解密是否成功“ 作为判断条件（比如开头应该是`308202`，e应该是`010001`，即65537）进行爆破：

<details>
    <summary>点击展开代码</summary>


```python
from binascii import a2b_base64, unhexlify
from Crypto.Hash import MD5
from Crypto.Cipher import DES3
from Crypto.Protocol.KDF import PBKDF1
import tqdm


def solve(data, salt):
    # 爆破一下passphrase
    for i in tqdm.trange(1000000):
        passphrase = str(i).zfill(6).encode()
        # We only support 3DES for encryption
        key = PBKDF1(passphrase, salt, 16, 1, MD5)
        key += PBKDF1(key + passphrase, salt, 8, 1, MD5)
        objenc = DES3.new(key, DES3.MODE_CBC, salt)
        # Encrypt with PKCS#7 padding
        data1 = objenc.decrypt(data).hex()
        if data1[:6] == "308202" and "010001" in data1:
            print("", data1)
            iv = a2b_base64(b"hQds7ZdA9yv+yKUYv2e4de8RxX356wYq7r8paBHPXisOkGIVEBYNviMSIbgelkSIjLQka+ZmC2YOgY/DgGJ82JmFG8mmYCcSooGL4ytVUY9dZa1khfhceg==")[:8]
            objenc = DES3.new(key, DES3.MODE_CBC, iv)
            c1 = a2b_base64(b"hQds7ZdA9yv+yKUYv2e4de8RxX356wYq7r8paBHPXisOkGIVEBYNviMSIbgelkSIjLQka+ZmC2YOgY/DgGJ82JmFG8mmYCcSooGL4ytVUY9dZa1khfhceg==")
            data1 = objenc.decrypt(c1)[8:].hex()
            print(data1)
            return passphrase


salt = unhexlify("435BF84C562FE793")
c = a2b_base64(b"9phAgeyjnJYZ6lgLYflgduBQjdX+V/Ph/fO8QB2ZubhBVOFJMHbwHbtgBaN3eGlhWiEFEdQWoOFvpip0whr4r7aGOhavWhIfRjiqfQVcKZx4/f02W4pcWVYo9/p3otdDig+kofIR9Ky8o9vQk7H1eESNMdq3PPmvd7KTE98ZPqtIIrjbSsJ9XRL+gr5a91gH")
passphrase = solve(c, salt)
print(passphrase)
"""
48%|████████████████████████████████████████████████████████████████████████▊                                                                              | 482292/1000000 [00:33<00:35, 14407.19it/s] 

前半：
3082025c02010002818100a18f011bebacceda1c6812730b9e62720d3cbd6857af2cf8431860f5dc83c5520f242f3be7c9e96d7f96b41898ff000fdb7e43ef6f1e717b2b7900f35660a21d1b16b51849be97a0b0f7cbcf5cfe0f00370cce6193fefa1fed97b37bd367a673565162ce17b0225708c032961d175bbc2c829bf2e16eabc7e0881feca0975c810203010001

后半：
8f2363b340e502405f152c429871a7acdd28be1b643b4652800b88a3d23cc57477d75dd5555b635167616ef5c609d69ce3c2aedcb03b62f929bbcd891cadc0ba031ae6fec8a2116d0808080808080808

b'483584'
"""
```

</details>

然后稍微整理一下（不懂的话，可以在网上搜一下**RSA私钥文件格式**去试着自己拆解一下），可以得到这些信息：

```python
n = 0x00a18f011bebacceda1c6812730b9e62720d3cbd6857af2cf8431860f5dc83c5520f242f3be7c9e96d7f96b41898ff000fdb7e43ef6f1e717b2b7900f35660a21d1b16b51849be97a0b0f7cbcf5cfe0f00370cce6193fefa1fed97b37bd367a673565162ce17b0225708c032961d175bbc2c829bf2e16eabc7e0881feca0975c81
e = 0x010001

dql = 0x6a033064c5a0dffc8f2363b340e5
qp = 0x5f152c429871a7acdd28be1b643b4652800b88a3d23cc57477d75dd5555b635167616ef5c609d69ce3c2aedcb03b62f929bbcd891cadc0ba031ae6fec8a2116d
```

其中，dql只有48bit（因为CBC模式的特点，所以后半部分的前8个bytes我们是没法解密的）

而我们可以知道的是：
$$
e*dq=1+k*(q-1)~\rightarrow k*q=edq+k-1
$$

$$
inv(q,p)*q\equiv1~(mod~p)\rightarrow inv(q,p)*q-1\equiv0~(mod~p)
$$

不过这里有 $k$ 和 $q$ 两个未知量，不过我们知道dql的值，所以我们有下面这样的方程（主要是为了减少要设的未知量）：
$$
X=k*q=e(x+dql)+k-1
$$

$$
inv(q,p)*(k*q)^2-k*(k*q)=inv(q,p)*X^2-k*X\equiv0~(mod~p)
$$

于是我们爆破k，解个copper就行（但比赛的时候调小了。。。问了Dexter师傅才知道得调大点才行）

exp：

<details>
    <summary>点击展开代码</summary>


```python
from tqdm import *
from Crypto.Util.number import *
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

n = 0xa18f011bebacceda1c6812730b9e62720d3cbd6857af2cf8431860f5dc83c5520f242f3be7c9e96d7f96b41898ff000fdb7e43ef6f1e717b2b7900f35660a21d1b16b51849be97a0b0f7cbcf5cfe0f00370cce6193fefa1fed97b37bd367a673565162ce17b0225708c032961d175bbc2c829bf2e16eabc7e0881feca0975c81
e = 65537
dq_leak= 0x8f2363b340e5
inv = 0x5f152c429871a7acdd28be1b643b4652800b88a3d23cc57477d75dd5555b635167616ef5c609d69ce3c2aedcb03b62f929bbcd891cadc0ba031ae6fec8a2116d
c = 55149764057291700808946379593274733093556529902852874590948688362865310469901900909075397929997623185589518643636792828743516623112272635512151466304164301360740002369759704802706396320622342771513106879732891498365431042081036698760861996177532930798842690295051476263556258192509634233232717503575429327989

def coppersmith(k):
    R.<x> = PolynomialRing(Zmod(n))
    tmp = e * (x * 2^48 + dq_leak) + k - 1
    f = inv * tmp^2 - k*tmp
    f = f.monic()
    x0 = f.small_roots(X=2^464,beta=1,epsilon=0.09)
    return x0

for k in trange(1,e):
    x0 = coppersmith(k)
    if x0 != []:
        dq = int(x0[0]) * 2^48 + dq_leak
        q = (e*dq + k - 1) // k
        p = n // q
        d = inverse(e,(p-1)*(q-1))
        m = pow(c,d,n)
        print(long_to_bytes(int(m)))
        break
"""
73%|███████▎  | 47793/65536 [11:46<04:22, 67.62it/s]

b'flag{df4a4054-23eb-4ba4-be5e-15b247d7b819}'
"""
```

</details>

---

## Hash（复现）

**题目：**

<details>
    <summary>点击展开代码</summary>


```python
#!/usr/bin/python2
# Python 2.7 (64-bit version)
from secret import flag
import os, binascii, hashlib
key = os.urandom(7)
print hash(key)
print int(hashlib.sha384(binascii.hexlify(key)).hexdigest(), 16) ^ int(binascii.hexlify(flag), 16)
"""
7457312583301101235
13903983817893117249931704406959869971132956255130487015289848690577655239262013033618370827749581909492660806312017
"""
```

</details>

### 1，MITM

题目使用了python2.7版本的hash()函数，所以查看一下[源码](https://github.com/neuml/py27hash/blob/master/src/python/py27hash/hash.py))，然后我们找到对应的hash代码（这里要注意——py2.7里，str和bytes不区分，所以在py2.7上 type(bytes型数据) 的返回值是str）：

<details>
    <summary>点击展开代码</summary>


```python
	def shash(value):
        """
        Returns a Python 2.7 hash for a string.

        Logic ported from the 2.7 Python branch: cpython/Objects/stringobject.c
        Method: static long string_hash(PyStringObject *a)

        Args:
            value: input string

        Returns:
            Python 2.7 hash
        """

        length = len(value)

        if length == 0:
            return 0

        mask = 0xffffffffffffffff
        x = (Hash.ordinal(value[0]) << 7) & mask
        for c in value:
            x = (1000003 * x) & mask ^ Hash.ordinal(c)

        x ^= length & mask

        # Convert to C long type
        x = ctypes.c_long(x).value

        if x == -1:
            x = -2

        return x

    @staticmethod
    def ordinal(value):
        """
        Converts value to an ordinal or returns the input value if it's an int.

        Args:
            value: input

        Returns:
            ordinal for value
        """

        return value if isinstance(value, int) else ord(value)
```

</details>

其实上述代码只是进行了这样的过程：

> mask = 0xffffffffffffffff
>
> 1，x = (Hash.ordinal(value[0]) << 7) & mask
>
> 2，重复**len(value)**次：**x = (1000003 * x) & mask ^ Hash.ordinal(value[0])**
>
> 3，x ^= length & mask

所以我们可以整一下Meet-in-the-middle：**正算三轮的结果存成一个表，逆算四轮的结果去匹配一下，如果匹配上了就说明整对了**。

逆算的操作如下：

> **x = (x ^ Hash.ordinal(value[0]) * inv(1000003, mask+1)) & mask**

exp：

<details>
    <summary>点击展开代码</summary>


```python
from tqdm import *
from Crypto.Util.number import *
import hashlib

def attack(t):
    mask = 0xffffffffffffffff
    table = {}
    for v0 in trange(256):
        x0 = (v0 << 7) & mask
        x0 = (1000003 * x0) & mask ^ v0
        for v1 in range(256):
            x1 = x0
            x1 = (1000003 * x1) & mask ^ v1
            for v2 in range(256):
                x2 = x1
                x2 = (1000003 * x2) & mask ^ v2
                table[x2] = v0*256*256+v1*256+v2
    
    inv=inverse(1000003,mask+1)
    t=t^7
    for v0 in trange(256):
        x= ((t^v0)*inv) & mask
        for v1 in range(256):
            x1 = x
            x1 = ((x1^v1)*inv) & mask
            for v2 in range(256):
                x2 = x1
                x2 = ((x2^v2)*inv) & mask
                for v3 in range(256):
                    x3 = x2
                    x3 = ((x3^v3)*inv) & mask
                    try:
                        return (table[x3],v3,v2,v1,v0)
                        
                    except:
                        continue

t = 7457312583301101235
keys = attack(t)
key = b''
for i in keys:
    key += long_to_bytes(i)
print(key)
c = 13903983817893117249931704406959869971132956255130487015289848690577655239262013033618370827749581909492660806312017
import binascii
key = int(hashlib.sha384(binascii.hexlify(key)).hexdigest(), 16)
print(long_to_bytes(c ^ key))
"""
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 256/256 [00:06<00:00, 39.78it/s] 
 32%|████████████████████████████████████████▋                                                                                      | 82/256 [11:17<23:56,  8.26s/it] 
b']\x8c\xf0?Z\x08R'
b'flag{bdb537aa-87ef-4e95-bea4-2f79259bdd07}'
"""
```

</details>

### 2，Lattice

这个方法还是听鸡块师傅才知道的。

![image-20240519235534408](CISCN2024-Crypto/image-20240519235534408.png)

不过按自己的理解去造了下，没出结果（后面听别的佬说有类似的题，但这周没空接着整了，有空就接着写这部分）
