import argparse

def get_paramters():
    parser = argparse.ArgumentParser(description="MobileNetV2")
    parser.add_argument('--model_name', type=str, default='mobilenetv2')
    parser.add_argument('--num_samples', type=int, help='the number of train samples')
    parser.add_argument('--num_valid_samples', type=int, help='the number of test samples')
    parser.add_argument('--num_test_samples', type=int, help='the number of test samples')

    parser.add_argument('--epoch', type=int, default=500)  ######
    parser.add_argument('--batch_size', type=int, default=64)  #####
    parser.add_argument('--num_classes', type=int, default=5)  #####
    parser.add_argument('--height', type=int, default=64)  #####
    parser.add_argument('--width', type=int, default=64)  ######
    parser.add_argument('--channel', type=int, default=3)

    parser.add_argument('--learning_rate', type=float, default=0.01)  # 0.001
    parser.add_argument('--lr_decay', type=float, default=0.8685)  # 0.98
    parser.add_argument('--num_epochs_per_decay', type=int, default=4)  # 10
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.9)  # for adam
    parser.add_argument('--momentum', type=float, default=0.9)  # for momentum

    parser.add_argument('--save_model_per_num_epoch', type=int, default=10)
    parser.add_argument('--channel_rito', type=float, default=1)
    parser.add_argument('--dataset_dir', type=str, default='./tfrecords', help='tfrecord file dir')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--test_record_name', type=str, default='expression_dect_test.tfrecord')

    parser.add_argument('--test_model', type=str)
    # parser.add_argument('--gpu', dest='gpu' ,action='store_false')

    parser.add_argument('--renew', type=bool, default=False)
    parser.add_argument('--is_train', type=int)

    args = parser.parse_args()
    return args