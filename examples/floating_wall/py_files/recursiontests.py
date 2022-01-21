for child in self.children:
    numpy.append(local, func(child, acc = acc))
