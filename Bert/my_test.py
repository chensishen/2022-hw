import json
import numpy as np
import random
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(6666)

# 将 "fp16_training "改为 True，以支持自动混合精度训练 (fp16)
fp16_training = True

if fp16_training:
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

# 加载预训练模型和标记符（Tokenizer）
model = BertForQuestionAnswering.from_pretrained('bert-base-chinese').to(device)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')


def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]


train_questions, train_paragraphs = read_data("hw7_train.json")
dev_questions, dev_paragraphs = read_data("hw7_dev.json")
test_questions, test_paragraphs = read_data("hw7_test.json")

# 分别标记问题和段落
# 将 "add_special_tokens "设为 False，因为在数据集 __getitem__ 中合并标记化的问题和段落时，将添加特殊标记。

train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions],
                                      add_special_tokens=False)
dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions],
                                    add_special_tokens=False)
test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions],
                                     add_special_tokens=False)
train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)


class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 40
        self.max_paragraph_len = 150

        ##### TODO: 更改 doc_stride 的值 #####
        # 控制段落滑动窗口的步长，避免直接截断信息。
        self.doc_stride = 150

        # 输入序列长度 = [CLS] + 问题 + [SEP] + 段落 + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        if self.split == 'train':
            # 将答案在段落文字中的起始/结束位置转换为标记化段落中的起始/结束位置
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            # 将段落中包含答案的部分切成片段，就得到了一个窗口
            mid = (answer_start_token + answer_end_token) // 2
            paragraph_start = max(0, min(mid - self.max_paragraph_len // 2,
                                         len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len

            # 切分问题/段落并添加特殊标记（101：CLS，102：SEP）
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start: paragraph_end] + [102]

            # 将标记化段落中答案的起始/结束位置转换为窗口中的起始/结束位置
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start

            # 填充序列并获得模型输入
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
                attention_mask), answer_start_token, answer_end_token

        # 验证/测试
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []

            # 段落被分割成几个窗口，每个窗口的起始位置用步长 "doc_stride "隔开
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                # 切分问题/段落并添加特殊标记（101：CLS，102：SEP）
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i: i + self.max_paragraph_len] + [102]

                # 填充序列并获得模型输入
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)

                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)

            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # 如果序列长度小于 max_seq_len，则为零
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # 词汇表中输入序列标记的索引
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # 表示输入第一和第二部分的分段标记符号索引。索引在 [0, 1] 中选择
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # 屏蔽，以避免对填充标记索引执行关注。屏蔽值在 [0, 1] 中选择
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        return input_ids, token_type_ids, attention_mask

train_set = QA_Dataset('train', train_questions, train_questions_tokenized, train_paragraphs_tokenized)
dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)

train_batch_size = 32

# 注意：请勿更改 dev_loader / test_loader 的批次大小！
# 虽然批次大小=1，但它实际上是由同一质量保证对的多个窗口组成的批次
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

# 评估功能
def evaluate(data, output):
    answer = '' # 存储最终的预测答案。
    max_prob = float('-inf') # 记录当前最高的答案概率，用于选择最佳的答案。
    num_of_windows = data[0].shape[1]

    # 表示在一个 batch 中每个段落的窗口数量。循环遍历每个窗口
    for k in range(num_of_windows):
        # 通过选择最可能的起始位置/结束位置来获取答案
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        # 答案的概率计算为起始概率和结束概率之和
        prob = start_prob + end_prob

        # 如果计算出的概率大于之前的窗口，则替换答案
        if prob > max_prob:
            max_prob = prob
            # 将令牌转换为字符（例如 [1920, 7032] --> "大金"）。
            answer = tokenizer.decode(data[0][0][k][start_index:end_index+1])

    # 删除答案中的空格（如 "大金"-->"大金"）。
    return answer.replace(' ', '')

if __name__ == "__main__":
    #训练
    num_epoch = 1
    validation = True
    logging_step = 100
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if fp16_training:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()

    print("Start Training")

    for epoch in range(num_epoch):
        step = 1
        train_loss = train_acc = 0

        for data in tqdm(train_loader):
            # 将所有数据加载到 GPU
            data = [i.to(device) for i in data]

            # 模型输入：input_ids、token_type_ids、attention_mask、start_positions、end_positions（注意：只有 "input_ids "是必填项）
            # 模型输出：start_logits、end_logits、loss（提供 start_positions/end_positions 时返回）
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])

            # 选择最可能的起始位置/结束位置
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)

            # 计算当前批次的准确率，只有当 start_index 和 end_index 都正确时，预测才是正确的
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            train_loss += output.loss

            if fp16_training:
                accelerator.backward(output.loss)
            else:
                output.loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            step +=1

            ##### TODO: 应用线性学习率衰减 #####
            lr_this_step = learning_rate * (1 - (epoch * len(train_loader) + step) / (num_epoch * len(train_loader)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step

            # 打印过去记录步骤中的训练损耗和准确性
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0

        if validation:
            print("Evaluating Dev Set ...")
            model.eval()
            with torch.no_grad():
                dev_acc = 0
                for i, data in enumerate(tqdm(dev_loader)):
                    output = model(input_ids=data[0].squeeze(dim=0).to(device),
                                   token_type_ids=data[1].squeeze(dim=0).to(device),
                                   attention_mask=data[2].squeeze(dim=0).to(device))
                    # 只有当答案文本完全匹配时，预测才是正确的
                    dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
                print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")


    # 保存模型及其配置文件到目录 "saved_model
    # 即在 "saved_model "目录下有两个文件： "pytorch_model.bin "和 "config.json"。
    # 可使用「model = BertForQuestionAnswering.from_pretrained("saved_model")」重新加载保存的模型。

    print("Saving Model ...")
    model_save_dir = "saved_model"
    model.save_pretrained(model_save_dir)

    # 测试
    print("Evaluating Test Set ...")

    result = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                           attention_mask=data[2].squeeze(dim=0).to(device))
            result.append(evaluate(data, output))

    result_file = "result.csv"
    with open(result_file, 'w') as f:
        f.write("ID,Answer\n")
        for i, test_question in enumerate(test_questions):
            # 用空字符串替换答案中的逗号（因为 csv 用逗号分隔）
            # 以相同方式处理 kaggle 中的答案
            f.write(f"{test_question['id']},{result[i].replace(',','')}\n")

    print(f"Completed! Result is in {result_file}")