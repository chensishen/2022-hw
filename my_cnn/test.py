from my_CNN import *

if __name__ == '__main__':

    model_best = Classifier().to(device)
    model_best.load_state_dict(torch.load(config['resnet_save_path']+'.ckpt',weights_only=False))
    model_best.eval()
    # prediction = []
    # with torch.no_grad():
    #     for data, _ in test_loader:
    #         test_pred = model_best(data.to(device))
    #         test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
    #         prediction += test_label.squeeze().tolist()

    test_loaders = [test_loader_extra1, test_loader_extra2, test_loader_extra3, test_loader]
    loader_nums = len(test_loaders)
    # 存储每个dataloader预测结果，一个dataloader一个数组
    loader_pred_list = []
    for idx, d_loader in enumerate(test_loaders):
        # 存储一个dataloader的预测结果,  一个batch一个是数组
        pred_arr_list = []
        with torch.no_grad():
            tq_bar = tqdm(d_loader)
            tq_bar.set_description(f"[ DataLoader {idx + 1}/{loader_nums} ]")
            for data, _ in tq_bar:
                test_pred = model_best(data.to(device))
                logit_pred = test_pred.cpu().data.numpy()
                pred_arr_list.append(logit_pred)
            # 将每个batch的预测结果合并成一个数组
            loader_pred_list.append(np.concatenate(pred_arr_list, axis=0))
    # 将预测结果合并
    pred_arr = np.zeros(loader_pred_list[0].shape)
    for pred_arr_t in loader_pred_list:
        pred_arr += pred_arr_t
    soft_vote_prediction = np.argmax(0.5 * pred_arr / len(loader_pred_list)
                                     + 0.5 * loader_pred_list[-1], axis=1)

    df = pd.DataFrame()
    # 保证ID为四位数（前面填充0）
    df["Id"] = [str(i).zfill(4) for i in range(1, len(test_set) + 1)]
    df["Category"] = soft_vote_prediction
    df.to_csv("submission.csv", index=False)