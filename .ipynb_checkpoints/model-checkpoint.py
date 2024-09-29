import torch
import torch.nn as nn
from config.bodyscan_config import model_params
import SetTransformer_Extrapolating as st
from PCT.model import Pct

# Fine-tuning model class
class FineTuneModel(nn.Module):
    def __init__(self, pretrained_model, additional_layers, final_layer):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.additional_layers = additional_layers
        self.final_layer = final_layer
    
    def forward(self, inputs, get_embeddings=True, get_embeddings_additional_layer=False):
        _, outputs = self.pretrained_model(inputs, get_embeddings=get_embeddings)
        embeddings = self.additional_layers(outputs)
        outputs = self.final_layer(embeddings)
        if get_embeddings_additional_layer:
            return outputs, embeddings
        return outputs
    
def get_pretrained_layers_stack(pretrained_model, projection_dim, num_classes):
    # Additional layers for pretrained model
        additional_layers = nn.Sequential(
            nn.Linear(pretrained_model.embed_dim, projection_dim), 
            nn.LeakyReLU(),
            nn.Dropout(p=0.1)
        )

        # Final layer 
        final_layer = nn.Sequential(
            nn.Linear((projection_dim), num_classes)
        )

        # Stack all layers
        model = FineTuneModel(
            pretrained_model,
            additional_layers,
            final_layer
        )
        return model

# Create Set Transformer model (either new or pretrained)
def create_model_st(num_classes, pretrained_path=None):
    model = None
    file_name = ''
    if pretrained_path:
        print('*************** Using pretrained '+ model_params['model'] +' ***************')
        pretrained_model = torch.load(pretrained_path)
        projection_dim = model_params['projection_dim']
        model = get_pretrained_layers_stack(pretrained_model, projection_dim, num_classes)
        file_name = pretrained_path.split('/')[-1]
    
    else: 
        print('*************** New model '+ model_params['model'] +' ***************')
        
        model = st.PyTorchModel(
            embed_dim=model_params['embed_dim'],
            num_heads=model_params['num_heads'],
            num_induce=model_params['num_induce'],
            stack=model_params['stack'],
            ff_activation=model_params['ff_activation'],
            dropout=model_params['dropout'],
            use_layernorm=model_params['use_layernorm'],
            pre_layernorm=model_params['pre_layernorm'],
            is_final_block = model_params['is_final_block'],
            num_classes=num_classes
        )
        print('Details:', 'embed_dim =', model_params['embed_dim'],
        'num_heads =', model_params['num_heads'],
        'num_induce =', model_params['num_induce'],
        'stack =', model_params['stack'],
        'dropout =', model_params['dropout'])

        file_name = str(model_params['embed_dim'])+'_'+str(model_params['num_heads'])+'_'+str(model_params['num_induce'])+'_'+str(model_params['stack'])+'_'+str(model_params['dropout'])
    return model, file_name
    

# Create PCT Model
def create_model_pct(num_classes, pretrained_path=None):
    model = None
    file_name = ''
    if pretrained_path:
        print('*************** Using pretrained '+ model_params['model'] +' ***************')
        pretrained_model = torch.load(pretrained_path)
        projection_dim = model_params['projection_dim']
        model = get_pretrained_layers_stack(pretrained_model, projection_dim, num_classes)
        file_name = pretrained_path.split('/')[-1]
    else:
        print('*************** New model '+ model_params['model'] +' ***************')

        dropout = model_params['dropout']
        embed_dim = model_params['embed_dim']
        model = Pct(dropout, output_channels = num_classes)

        print('Details:', 'embed_dim =', embed_dim,
        'dropout =', dropout)
        
        # To keep unique for the output files
        file_name = 'ed'+str(embed_dim)+'_'+str(dropout)
    return model, file_name    

