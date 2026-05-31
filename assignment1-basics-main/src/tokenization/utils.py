def bytes_to_unicode():
        """
        chr()
        ord()
        """
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("?"), ord("?") + 1)) + list(range(ord("?"), ord("?") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))