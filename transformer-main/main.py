# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch.nn as nn
import torch.optim as optim
from datasets import *
from transformer import Transformer

if __name__ == "__main__":

    enc_inputs, dec_inputs, dec_outputs = make_data()
    print(enc_inputs.shape)
    print(dec_inputs.shape)
    print(dec_outputs.shape)

    print(enc_inputs)
    print(dec_inputs)
    print(dec_outputs)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

    for data in loader:
        print(type(data))
        print(len(data))
        print(data)

    model = Transformer().cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)         # 忽略 占位符 索引为0.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)


    for epoch in range(50):
        for enc_inputs, dec_inputs, dec_outputs in loader:  # enc_inputs : [batch_size, src_len]
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model, 'model.pth')
    print("保存模型")
