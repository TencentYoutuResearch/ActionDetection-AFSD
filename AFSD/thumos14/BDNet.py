import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from AFSD.common.i3d_backbone import InceptionI3d
from AFSD.common.config import config
from AFSD.common.layers import Unit1D, Unit3D
from AFSD.prop_pooling.boundary_pooling_op import BoundaryMaxPooling

num_classes = config['dataset']['num_classes']
freeze_bn = config['model']['freeze_bn']
freeze_bn_affine = config['model']['freeze_bn_affine']

layer_num = 6
conv_channels = 512
feat_t = 256 // 4


class I3D_BackBone(nn.Module):
    def __init__(self, final_endpoint='Mixed_5c', name='inception_i3d', in_channels=3,
                 freeze_bn=freeze_bn, freeze_bn_affine=freeze_bn_affine):
        super(I3D_BackBone, self).__init__()
        self._model = InceptionI3d(final_endpoint=final_endpoint,
                                   name=name,
                                   in_channels=in_channels)
        self._model.build()
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

    def load_pretrained_weight(self, model_path='models/i3d_models/rgb_imagenet.pt'):
        self._model.load_state_dict(torch.load(model_path), strict=False)

    def train(self, mode=True):
        super(I3D_BackBone, self).train(mode)
        if self._freeze_bn and mode:
            # print('freeze all BatchNorm3d in I3D backbone.')
            for name, m in self._model.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    # print('freeze {}.'.format(name))
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def forward(self, x):
        return self._model.extract_features(x)


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return torch.exp(input * self.scale)


class ProposalBranch(nn.Module):
    def __init__(self, in_channels, proposal_channels):
        super(ProposalBranch, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.lr_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels * 2,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.boundary_max_pooling = BoundaryMaxPooling()

        self.roi_conv = nn.Sequential(
            Unit1D(in_channels=proposal_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.proposal_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels * 4,
                output_channels=in_channels,
                kernel_shape=1,
                activation_fn=None
            ),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, frame_level_feature, segments, frame_segments):
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)
        prop_feature = self.boundary_max_pooling(feature, segments)
        prop_roi_feature = self.boundary_max_pooling(frame_level_feature, frame_segments)
        prop_roi_feature = self.roi_conv(prop_roi_feature)
        prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short], dim=1)
        prop_feature = self.proposal_conv(prop_feature)
        return prop_feature, feature


class CoarsePyramid(nn.Module):
    def __init__(self, feat_channels, frame_num=256):
        super(CoarsePyramid, self).__init__()
        out_channels = conv_channels
        self.pyramids = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        self.frame_num = frame_num
        self.layer_num = layer_num
        self.pyramids.append(nn.Sequential(
            Unit3D(
                in_channels=feat_channels[0],
                output_channels=out_channels,
                kernel_shape=[1, 6, 6],
                padding='spatial_valid',
                use_batch_norm=False,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))

        self.pyramids.append(nn.Sequential(
            Unit3D(
                in_channels=feat_channels[1],
                output_channels=out_channels,
                kernel_shape=[1, 3, 3],
                use_batch_norm=False,
                padding='spatial_valid',
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))
        for i in range(2, layer_num):
            self.pyramids.append(nn.Sequential(
                Unit1D(
                    in_channels=out_channels,
                    output_channels=out_channels,
                    kernel_shape=3,
                    stride=2,
                    use_bias=True,
                    activation_fn=None
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))

        loc_towers = []
        for i in range(2):
            loc_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.loc_tower = nn.Sequential(*loc_towers)
        conf_towers = []
        for i in range(2):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.conf_tower = nn.Sequential(*conf_towers)

        self.loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )
        self.conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.loc_proposal_branch = ProposalBranch(out_channels, 512)
        self.conf_proposal_branch = ProposalBranch(out_channels, 512)

        self.prop_loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=1,
            activation_fn=None
        )
        self.prop_conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=1,
            activation_fn=None
        )

        self.center_head = Unit1D(
            in_channels=out_channels,
            output_channels=1,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.deconv = nn.Sequential(
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 1, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

        self.priors = []
        t = feat_t
        for i in range(layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2

    def forward(self, feat_dict, ssl=False):
        pyramid_feats = []
        locs = []
        confs = []
        centers = []
        prop_locs = []
        prop_confs = []
        trip = []
        x2 = feat_dict['Mixed_5c']
        x1 = feat_dict['Mixed_4f']
        batch_num = x1.size(0)
        for i, conv in enumerate(self.pyramids):
            if i == 0:
                x = conv(x1)
                x = x.squeeze(-1).squeeze(-1)
            elif i == 1:
                x = conv(x2)
                x = x.squeeze(-1).squeeze(-1)
                x0 = pyramid_feats[-1]
                y = F.interpolate(x, x0.size()[2:], mode='nearest')
                pyramid_feats[-1] = x0 + y
            else:
                x = conv(x)
            pyramid_feats.append(x)

        frame_level_feat = pyramid_feats[0].unsqueeze(-1)
        frame_level_feat = F.interpolate(frame_level_feat, [self.frame_num, 1]).squeeze(-1)
        frame_level_feat = self.deconv(frame_level_feat)
        trip.append(frame_level_feat.clone())
        start_feat = frame_level_feat[:, :256]
        end_feat = frame_level_feat[:, 256:]
        start = start_feat.permute(0, 2, 1).contiguous()
        end = end_feat.permute(0, 2, 1).contiguous()

        for i, feat in enumerate(pyramid_feats):
            loc_feat = self.loc_tower(feat)
            conf_feat = self.conf_tower(feat)

            locs.append(
                self.loc_heads[i](self.loc_head(loc_feat))
                    .view(batch_num, 2, -1)
                    .permute(0, 2, 1).contiguous()
            )
            confs.append(
                self.conf_head(conf_feat).view(batch_num, num_classes, -1)
                    .permute(0, 2, 1).contiguous()
            )
            t = feat.size(2)
            with torch.no_grad():
                segments = locs[-1] / self.frame_num * t
                priors = self.priors[i].expand(batch_num, t, 1).to(feat.device)
                new_priors = torch.round(priors * t - 0.5)
                plen = segments[:, :, :1] + segments[:, :, 1:]
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)

                l_segment = new_priors - segments[:, :, :1]
                r_segment = new_priors + segments[:, :, 1:]
                segments = torch.cat([
                    torch.round(l_segment - out_plen),
                    torch.round(l_segment + in_plen),
                    torch.round(r_segment - in_plen),
                    torch.round(r_segment + out_plen)
                ], dim=-1)

                decoded_segments = torch.cat(
                    [priors[:, :, :1] * self.frame_num - locs[-1][:, :, :1],
                     priors[:, :, :1] * self.frame_num + locs[-1][:, :, 1:]],
                    dim=-1)
                plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)
                frame_segments = torch.cat([
                    torch.round(decoded_segments[:, :, :1] - out_plen),
                    torch.round(decoded_segments[:, :, :1] + in_plen),
                    torch.round(decoded_segments[:, :, 1:] - in_plen),
                    torch.round(decoded_segments[:, :, 1:] + out_plen)
                ], dim=-1)

            loc_prop_feat, loc_prop_feat_ = self.loc_proposal_branch(loc_feat, frame_level_feat,
                                                                     segments, frame_segments)
            conf_prop_feat, conf_prop_feat_ = self.conf_proposal_branch(conf_feat, frame_level_feat,
                                                                        segments, frame_segments)
            if i == 0:
                trip.extend([loc_prop_feat_.clone(), conf_prop_feat_.clone()])
                ndim = loc_prop_feat_.size(1) // 2
                start_loc_prop = loc_prop_feat_[:, :ndim, ].permute(0, 2, 1).contiguous()
                end_loc_prop = loc_prop_feat_[:, ndim:, ].permute(0, 2, 1).contiguous()
                start_conf_prop = conf_prop_feat_[:, :ndim, ].permute(0, 2, 1).contiguous()
                end_conf_prop = conf_prop_feat_[:, ndim:, ].permute(0, 2, 1).contiguous()
                if ssl:
                    return trip
            prop_locs.append(self.prop_loc_head(loc_prop_feat).view(batch_num, 2, -1)
                             .permute(0, 2, 1).contiguous())
            prop_confs.append(self.prop_conf_head(conf_prop_feat).view(batch_num, num_classes, -1)
                              .permute(0, 2, 1).contiguous())
            centers.append(
                self.center_head(loc_prop_feat).view(batch_num, 1, -1)
                    .permute(0, 2, 1).contiguous()
            )

        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
        conf = torch.cat([o.view(batch_num, -1, num_classes) for o in confs], 1)
        prop_loc = torch.cat([o.view(batch_num, -1, 2) for o in prop_locs], 1)
        prop_conf = torch.cat([o.view(batch_num, -1, num_classes) for o in prop_confs], 1)
        center = torch.cat([o.view(batch_num, -1, 1) for o in centers], 1)
        priors = torch.cat(self.priors, 0).to(loc.device)
        return loc, conf, prop_loc, prop_conf, center, priors, start, end, \
               start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop


class BDNet(nn.Module):
    def __init__(self, in_channels=3, backbone_model=None, training=True):
        super(BDNet, self).__init__()

        self.coarse_pyramid_detection = CoarsePyramid([832, 1024])
        self.reset_params()

        self.backbone = I3D_BackBone(in_channels=in_channels)
        self.boundary_max_pooling = BoundaryMaxPooling()
        self._training = training

        if self._training:
            if backbone_model is None:
                self.backbone.load_pretrained_weight()
            else:
                self.backbone.load_pretrained_weight(backbone_model)
        self.scales = [1, 4, 4]

    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1., (fan_in + fan_out) / 2.)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose3d):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, proposals=None, ssl=False):
        # x should be [B, C, 256, 96, 96] for THUMOS14
        feat_dict = self.backbone(x)
        if ssl:
            top_feat = self.coarse_pyramid_detection(feat_dict, ssl)
            decoded_segments = proposals[0].unsqueeze(0)
            plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
            in_plen = torch.clamp(plen / 4.0, min=1.0)
            out_plen = torch.clamp(plen / 10.0, min=1.0)
            frame_segments = torch.cat([
                torch.round(decoded_segments[:, :, :1] - out_plen),
                torch.round(decoded_segments[:, :, :1] + in_plen),
                torch.round(decoded_segments[:, :, 1:] - in_plen),
                torch.round(decoded_segments[:, :, 1:] + out_plen)
            ], dim=-1)
            anchor, positive, negative = [], [], []
            for i in range(3):
                bound_feat = self.boundary_max_pooling(top_feat[i], frame_segments / self.scales[i])
                # for triplet loss
                ndim = bound_feat.size(1) // 2
                anchor.append(bound_feat[:, ndim:, 0])
                positive.append(bound_feat[:, :ndim, 1])
                negative.append(bound_feat[:, :ndim, 2])

            return anchor, positive, negative
        else:
            loc, conf, prop_loc, prop_conf, center, priors, start, end, \
            start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop = \
                self.coarse_pyramid_detection(feat_dict)
            return {
                'loc': loc,
                'conf': conf,
                'priors': priors,
                'prop_loc': prop_loc,
                'prop_conf': prop_conf,
                'center': center,
                'start': start,
                'end': end,
                'start_loc_prop': start_loc_prop,
                'end_loc_prop': end_loc_prop,
                'start_conf_prop': start_conf_prop,
                'end_conf_prop': end_conf_prop
            }


def test_inference(repeats=3, clip_frames=256):
    model = BDNet(training=False)
    model.eval()
    model.cuda()
    import time
    run_times = []
    x = torch.randn([1, 3, clip_frames, 96, 96]).cuda()
    warmup_times = 2
    for i in range(repeats + warmup_times):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        run_times.append(time.time() - start)

    infer_time = np.mean(run_times[warmup_times:])
    infer_fps = clip_frames * (1. / infer_time)
    print('inference time (ms):', infer_time * 1000)
    print('infer_fps:', int(infer_fps))
    # print(y['loc'].size(), y['conf'].size(), y['priors'].size())


if __name__ == '__main__':
    test_inference(20, 256)
