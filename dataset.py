import torch.utils.data as Data
from transformers import AutoTokenizer

def replace_full_width_numbers(text):
    full_width_numbers = '０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
    half_width_numbers = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    trans_table = str.maketrans(full_width_numbers, half_width_numbers)
    return text.translate(trans_table)


def split_numbers(text):
    return re.sub(r"[0-9a-zA-Z○×]", lambda x: ' ' + x.group() + ' ', text)


class Bert_Dataset(Data.Dataset):
    def __init__(self, filename, lable2id, encoding="gbk", max_seq_len=512):
        self.max_seq_len = max_seq_len
        raw_texts, raw_labels = self.transfer_dataset(filename, encoding)
        self.labels = [[lable2id[element] for element in row] for row in raw_labels]

        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        raw_texts = [split_numbers(text) for text in raw_texts]
        self.inputs = tokenizer(raw_texts, max_length=max_seq_len, is_split_into_words=False, return_length=True)
        self.len = len(self.labels)

        assert all(len(row1) == len(row2) for row1, row2 in zip(self.inputs["input_ids"], self.labels))
        assert all(len(row1) == len(row2) for row1, row2 in zip(self.inputs["token_type_ids"], self.labels))


    def transfer_dataset(self, file_path, encoding):
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()

        bmes_data = []
        bmes_label = []

        pattern = re.compile(r'\d{8}-\d{2}-\d{3}-\d{3}/\w')

        for line in lines:
            line = pattern.sub('', line)
            line = replace_full_width_numbers(line)
            words = line.split()
            words = [word.split('/')[0] for word in words]
            sentence = "".join(words)
            bmes_data.append(sentence)

            line_label = []
            line_label.append("0")
            for word in words:
                word = word.split('/')[0]
                if len(word) == 1:
                    line_label.append('S')
                else:
                    line_label.append('B')
                    for _ in word[1:-1]:
                        line_label.append('M')
                    line_label.append('E')

            line_label = line_label[:self.max_seq_len-1]
            line_label.append("0")
            bmes_label.append(line_label)

        return bmes_data, bmes_label

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.inputs["input_ids"][index], self.inputs['token_type_ids'][index], self.labels[index]


if __name__ == "__main__":
    label2id = {"0":0, "B":1, "M":2, "E":3, "S":4}
    data = Bert_Dataset("ChineseCorpus199801.txt", label2id)
    print(data[0])