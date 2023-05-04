from gc import freeze
import math, torch
import torch.nn as nn
import torch.nn.functional as F


ONE_HOT = 1
POSITION = 2
FLOAT = 3
FORWARD = 4
PASS = 5
MAPPING = 6

class Feature():
    def __init__(self, type, out_dim, global_normalize, name):
        self.type = type
        self.out_dim = out_dim
        self.global_normalize = global_normalize
        self.name = name
        self.Active = True

class PositionFeature(Feature):
    def __init__(self, out_dim, name, max_len=500, global_normalize=False):
        super(PositionFeature, self).__init__(POSITION, out_dim, global_normalize, name)
        self.max_len = max_len
    
class OneHotFeature(Feature):
    def __init__(self, out_dim, n_classes, name, global_normalize=False):
        super(OneHotFeature, self).__init__(ONE_HOT, out_dim, global_normalize, name)
        self.n_classes = n_classes

class FloatFeature(Feature):
    def __init__(self, out_dim, name, global_normalize=False):
        super(FloatFeature, self).__init__(FLOAT, out_dim, global_normalize, name)

class ForwardFeature(Feature):
    def __init__(self, out_dim, in_dim, name, global_normalize=False):
        super(ForwardFeature, self).__init__(FORWARD, out_dim, global_normalize, name)
        self.in_dim = in_dim

class PassFeature(Feature):
    def __init__(self, dim, name, global_normalize=False):
        super(PassFeature, self).__init__(PASS, dim, global_normalize, name)
        self.in_dim = dim

class MappingFeature(Feature):
    def __init__(self, out_dim, name, global_normalize=False):
        super(MappingFeature, self).__init__(MAPPING, out_dim, global_normalize, name)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, poses):
        return self.pe[poses.long(), :]

class OneHotEncoding(nn.Module):
    def __init__(self, n_classes, dim_emb):
        super(OneHotEncoding, self).__init__()

        oe = F.one_hot(torch.arange(0,n_classes).long(), num_classes=n_classes).float()
        self.linear = nn.Linear(n_classes, dim_emb)
        self.register_buffer('oe', oe)

    
    def forward(self, poses):
        res = self.oe[poses.long(), :]
        res = self.linear(res)
        res = F.relu(res)
        return res

class FloatEncoding(nn.Module):
    def __init__(self,  dim_emb):
        super(FloatEncoding, self).__init__()
        self.linear = nn.Linear(1, dim_emb)
    
    def forward(self, input):
        return F.relu(self.linear(input.float()))
        #return self.linear(input.float())
        #return input.float()

class PassEncoding(nn.Module):
    def __init__(self):
        super(PassEncoding, self).__init__()
    def forward(self, input):
        return input.float()

class ForwardEncoding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ForwardEncoding, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, input):
        return F.relu(self.linear(input.float()))
        #return input.float()

class MappingEncoding(nn.Module):
    def __init__(self, pretrained, dev, dev1, dev2, dev3, dev4, dev5, dev6, dev7, freeze=False):
        super(MappingEncoding, self).__init__()
        pretrained_1 = pretrained[:, :44]
        pretrained_2 = pretrained[:, 44:88]
        pretrained_3 = pretrained[:, 88:132]
        pretrained_4 = pretrained[:, 132:177]
        pretrained_5 = pretrained[:, 177:221]
        pretrained_6 = pretrained[:, 221:255]
        pretrained_7 = pretrained[:, 255:]
        print(pretrained_1.shape)
        self.emb1 = nn.Embedding.from_pretrained(pretrained_1, freeze=freeze).to(dev1)
        self.emb2 = nn.Embedding.from_pretrained(pretrained_2, freeze=freeze).to(dev2)
        self.emb3 = nn.Embedding.from_pretrained(pretrained_3, freeze=freeze).to(dev3)
        self.emb4 = nn.Embedding.from_pretrained(pretrained_4, freeze=freeze).to(dev4)
        self.emb5 = nn.Embedding.from_pretrained(pretrained_5, freeze=freeze).to(dev5)
        self.emb6 = nn.Embedding.from_pretrained(pretrained_6, freeze=freeze).to(dev6)
        self.emb7 = nn.Embedding.from_pretrained(pretrained_7, freeze=freeze).to(dev7)
        self.embs = [self.emb1, self.emb2, self.emb3, self.emb4, self.emb5, self.emb6, self.emb7]
        self.dev = dev
        self.devs = [dev1, dev2, dev3, dev4, dev5, dev6, dev7]
    
    def forward(self, poses):
        if hasattr(self, 'devs'):
            poseses = [poses.to(dev) for dev in self.devs]
            reses = [self.embs[i](poseses[i].long()).to(self.dev) for i in range(len(poseses))]

            return torch.cat(tuple(reses), dim=1)
        else:
            return self.emb(poses.long())


#class MappingEncoding(nn.Module):
#    def __init__(self, pretrained, dev, dev1, dev2, dev3, dev4, dev5, dev6, dev7, freeze=False):
#        super(MappingEncoding, self).__init__()
#        pretrained_1 = pretrained[:, :33]
#        pretrained_2 = pretrained[:, 33:66]
#        pretrained_3 = pretrained[:, 66:]
#        print(pretrained_1.shape)
#        self.emb1 = nn.Embedding.from_pretrained(pretrained_1, freeze=freeze).to(dev1)
#        self.emb2 = nn.Embedding.from_pretrained(pretrained_2, freeze=freeze).to(dev2)
#        self.emb3 = nn.Embedding.from_pretrained(pretrained_3, freeze=freeze).to(dev3)
#        self.embs = [self.emb1, self.emb2, self.emb3]
#        self.dev = dev
#        self.devs = [dev1, dev2, dev3]
    
#    def forward(self, poses):
#        poseses = [poses.to(dev) for dev in self.devs]
#        reses = [self.embs[i](poseses[i].long()).to(self.dev) for i in range(len(poseses))]

#        return torch.cat(tuple(reses), dim=1)

class FeatureEncoding(nn.Module):

    def __init__(self, feature_types, word_vectors, dev, dev2, dev3, dev4, dev5, dev6, dev7, dev8):
        super(FeatureEncoding, self).__init__()
        self.layers = nn.ModuleList()
        self.feature_types = feature_types    
        for i,ft in enumerate(feature_types):
            if ft.type == ONE_HOT:
                self.layers.append(OneHotEncoding(ft.n_classes,  ft.out_dim).to(dev))
            if ft.type == POSITION:
                self.layers.append(PositionalEncoding(ft.out_dim, ft.max_len).to(dev))
            if ft.type == FLOAT:
                self.layers.append(FloatEncoding(ft.out_dim).to(dev))
            if ft.type == FORWARD:
                self.layers.append(ForwardEncoding(ft.in_dim, ft.out_dim).to(dev))
            if ft.type == PASS:
                self.layers.append(PassEncoding().to(dev))
            if ft.type == MAPPING:
                self.layers.append(MappingEncoding(word_vectors.pop(), dev, dev2, dev3, dev4, dev5, dev6, dev7, dev8))
    
    def forward(self, x, dev):
        tensors = []
        pos = 0
        for i,layer in enumerate(self.layers):
            if self.feature_types[i].type in [ONE_HOT, POSITION, MAPPING]:
                tensors.append(layer(x[:,pos]))
                pos += 1
            elif self.feature_types[i].type == FLOAT:
                z = x[:,pos]
                tensors.append(layer(z.unsqueeze(1)))
                pos += 1
            elif self.feature_types[i].type == FORWARD or self.feature_types[i].type == PASS:
                tensors.append(layer(x[:, pos:pos+self.feature_types[i].in_dim]))
                pos += self.feature_types[i].in_dim

        
        return torch.cat(tuple(tensors), dim=1)
