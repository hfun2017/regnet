import torch

from models.dgcnn import Encoder
from utils import *


class get_model(nn.Module):
    def __init__(self, d_model=128, channel=3, npoint=None):
        super(get_model, self).__init__()
        self.transformer = nn.Transformer(d_model, num_encoder_layers=3, num_decoder_layers=3)
        self.dgcnn=Encoder(d_model)
        self.pointnetDecoder = DispGenerator(d_model)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor, **kwargs):
        encoder_input, decoder_input = encoder_input.permute(0, 2, 1), decoder_input.permute(0, 2, 1)       #[B,3,N]
        embed_input = self.dgcnn(encoder_input)     # [B,D,N]
        embed_output = self.dgcnn(decoder_input)       # [B,D,N]
        global_feat = embed_output
        embed_input, embed_output = embed_input.permute(2, 0, 1), embed_output.permute(2, 0, 1)  # [N, B, d_model]
        transformer_out = self.transformer(embed_input, embed_output)  # [N, B ,d_model]
        transformer_out = transformer_out.permute(1, 0, 2)  # [B, N, d_model]

        warpped_feat = transformer_out      # [B,N,D]
        warpped_feat = torch.cat([warpped_feat, global_feat.permute(0, 2, 1)], dim=-1) #[B,N,2D]
        displacement = self.pointnetDecoder(warpped_feat.permute(0, 2, 1))  # [B,3,N]
        warped = displacement + decoder_input[:, 0:3, :]
        loss1 = chamfer_loss(encoder_input[:, :3, :], warped, ps=warped.size()[-1])

        return warped.permute(0, 2, 1), loss1  # , loss2


if __name__ == '__main__':
    model = get_model(128, 6)
    a = torch.rand((2, 1024, 6))
    b = torch.rand((2, 1024, 6))
    loss = model(a, b)
    print(loss)
