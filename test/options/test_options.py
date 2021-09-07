from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--warp_checkpoint', type=str, default='/home/sh0089/sen/PF-AFN/PF-AFN_train/checkpoints_ours_fc/PFAFN_e2e_ours/PFAFN_warp_epoch_101.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default='/home/sh0089/sen/PF-AFN/PF-AFN_train/checkpoints_ours_fc/PFAFN_e2e_ours/PFAFN_gen_epoch_101.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

        self.isTrain = False
