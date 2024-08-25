import torch
import os
import json
import time
import re
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import random

import nodes


class MagicalSaveNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {"images": ("IMAGE", ),
                    "Output_Path": ("STRING", {"default": '[time(%Y-%m-%d)]', "multiline": False, "tooltip":'Subfolder Path into "output"'}),
                    "Name": ("STRING", {"default": "ComfyUI", "tooltip":'File Name'}),
                    "Extension": (['png', 'jpg', 'tiff', 'bmp', 'none'],{"default":'png', "tooltip":'Image Type'}),
                    "Quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip":'jpg compression 1-100, png compression 0-9  (if > 9 = 0 lossless)'}),
                    "Save_gen_data_to_txt": (["true", "false"],{"default":"true", "tooltip":'True saves meta-data based on renamed nodes (right-click -> "Title") and the comfy-flow to a text file'}),
                    "Save_gen_data_to_png": (["true", "false"],{"default":"false", "tooltip":'True saves meta-data based on renamed nodes (right-click -> "Title") and the comfy-flow to a png image'}),
                },
               "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                
               }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True

    CATEGORY = "Hellrunner's"
    DESCRIPTION = 'Compiles meta-data based on renamed nodes (right-click -> "Title") and optionally includes it and the comfy-flow in a text file and/or a png image.'

    def save_images(self, images, Output_Path='[time(%Y-%m-%d)]', Name="ComfyUI", Extension='png', Quality=95, Save_gen_data_to_txt="true", Save_gen_data_to_png="false", prompt=None, extra_pnginfo=None):
        
        def replace_custom_time(match):
            format_code = match.group(1)
            return time.strftime(format_code, time.localtime(time.time()))

        def writeTextFile(file, content):
            try:
                with open(file, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(content)
            except OSError:
                print(str(f"Unable to save file `{file}`"))

        tokens = {'[time]': str(time.time())}

        tokens['[time]'] = str(time.time())
        if '.' in tokens['[time]']:
            tokens['[time]'] = tokens['[time]'].split('.')[0]

        for token, value in tokens.items():
            if token.startswith('[time('):
                continue
            Output_Path = Output_Path.replace(token, value)

        path = re.sub(r'\[time\((.*?)\)\]', replace_custom_time, Output_Path)

        full_output_folder = os.path.join(self.output_dir, path)
        if Name == "":
            Name="ComfyUI"
        filename = Name

        results = list()

        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder, exist_ok=True)

        List = os.listdir(full_output_folder)
        counter = 1
        for FileName in List:
            FileNameSplit = FileName.split(".")
            ext = FileNameSplit.pop()
            if ext == 'png' or ext == 'jpg' or ext == 'tiff' or ext == 'bmp':
                counter+=1
        
        file = f"{filename}_{counter:05}.{Extension}"
        
        while os.path.exists(os.path.join(full_output_folder, f"{filename}_{counter:05}.png")) or os.path.exists(os.path.join(full_output_folder, f"{filename}_{counter:05}.jpg")) or os.path.exists(os.path.join(full_output_folder, f"{filename}_{counter:05}.tiff")) or os.path.exists(os.path.join(full_output_folder, f"{filename}_{counter:05}.bmp")):
            counter += 1
            file = f"{filename}_{counter:05}.{Extension}"

        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            pngMeta = PngInfo()
            txtMeta = ""

            if prompt is not None:
                pngMeta.add_text("prompt", json.dumps(prompt))
            
            if extra_pnginfo is not None:
                for info in extra_pnginfo:
                    pngMeta.add_text(info, json.dumps(extra_pnginfo[info]))

                if extra_pnginfo["workflow"]:
                    if extra_pnginfo["workflow"]["nodes"]:
                        string = ""

                        for node in extra_pnginfo["workflow"]["nodes"]:
                            if 'title' in node: 
                                #print(node)
                                if node['type'] == "KSampler":
                                    string +=  f"{node['title']} - CFG scale: {node['widgets_values'][3]}, Steps: {node['widgets_values'][2]}, Sampler: {node['widgets_values'][4]} {node['widgets_values'][5]}, Denoise: {node['widgets_values'][6]}, Seed: {node['widgets_values'][0]}\n"
                                elif node['type'] == "KSamplerAdvanced":
                                    string +=  f"{node['title']} - CFG scale: {node['widgets_values'][4]}, Steps: {node['widgets_values'][3]}, Sampler: {node['widgets_values'][5]} {node['widgets_values'][6]}, Seed: {node['widgets_values'][1]}\n"
                                elif node['type'] == "STRING":
                                    string +=  f"{node['title']}: {node['widgets_values'][0]}\n"
                                elif node['type'] == "EmptyLatentImage":
                                    string += f"{node['title']} - Width: {node['widgets_values'][0]}, Height: {node['widgets_values'][1]}, Batch Size: {node['widgets_values'][2]}\n"
                                elif node['type'] == "ThermalLatenator":
                                    string += f"{node['title']} - Ratio Selected: {node['widgets_values'][0]}, Width Override: {node['widgets_values'][1]}, Height Override: {node['widgets_values'][2]}, Batch Count: {node['widgets_values'][3]}, Batch Size: {node['widgets_values'][4]}, First Seed: {node['widgets_values'][5]}, Batch Seeds: {node['widgets_values'][7]}\n"
                                elif node['type'] == "CheckpointLoaderSimple":
                                    string +=  f"{node['title']}: {node['widgets_values'][0]}\n"
                                elif node['type'] == "VAELoader":
                                    string +=  f"{node['title']}: {node['widgets_values'][0]}\n"
                                elif node['type'] == "LoraLoader":
                                    string += f"{node['title']} - LoRA Name: {node['widgets_values'][0]}, Model Strength: {node['widgets_values'][1]}, Text Encoder Strength: {node['widgets_values'][2]}\n"
                                else:
                                    if 'widgets_values' in node:
                                        string += f"{node['title']}: {node['widgets_values']}\n"                            

                        txtMeta += f"{string}\n"

                txtMeta += "Workflow: " + json.dumps(extra_pnginfo["workflow"]) + "\n"

            if Save_gen_data_to_txt == "true" :
                writeTextFile(os.path.join(full_output_folder, f"{filename}_{counter:05}.txt"), txtMeta)

            if Save_gen_data_to_png == "false":
                pngMeta=None

            print(os.path.join(full_output_folder, file))

            if Extension == 'png':
                if Quality>9:
                    Quality=0
                img.save(os.path.join(full_output_folder, file), pnginfo=pngMeta, compress_level=Quality, optimize=True)
            elif Extension == 'jpg':
                img.save(os.path.join(full_output_folder, file), quality=Quality, optimize=True)
            elif Extension == 'tiff':
                img.save(os.path.join(full_output_folder, file), compression=None, description=txtMeta)
            elif Extension == 'bmp':
                img.save(os.path.join(full_output_folder, file))

            results.append({
                "filename": file,
                "subfolder": path,
                "type": self.type
            })
            counter += 1
            file = f"{filename}_{counter:05}.{Extension}"
        
            while os.path.exists(os.path.join(full_output_folder, file)):
                counter += 1
                file = f"{filename}_{counter:05}.{Extension}"

        return { "ui": { "images": results } }


class thermalLatenator:

    @classmethod
    def INPUT_TYPES(s):
        s.ratio_dict = {
        "1:1 [1024x1024 square]": {"width": 1024, "height":  1024},
        "8:5 [1216x768 landscape]": {"width": 1216, "height":  768},
        "4:3 [1152x896 landscape]": {"width": 1152, "height":  896},
        "3:2 [1216x832 landscape]": {"width": 1216, "height":  832},
        "7:5 [1176x840 landscape]": {"width": 1176, "height":  840},
        "16:9 [1344x768 landscape]": {"width": 1344, "height":  768},
        "21:9 [1536x640 landscape]": {"width": 1536, "height":  640},
        "19:9 [1472x704 landscape]": {"width": 1472, "height":  704},
        "3:4 [896x1152 portrait]": {"width": 896, "height":  1152},
        "2:3 [832x1216 portrait]": {"width": 832, "height":  1216},
        "5:7 [840x1176 portrait]": {"width": 840, "height":  1176},
        "9:16 [768x1344 portrait]": {"width": 768, "height":  1344},
        "9:21 [640x1536 portrait]": {"width": 640, "height":  1536},
        "5:8 [768x1216 portrait]": {"width": 768, "height":  1216},
        "9:19 [704x1472 portrait]": {"width": 704, "height":  1472}
}
        s.ratio_sizes = list(s.ratio_dict.keys())
        default_ratio = s.ratio_sizes[0]

        return {"required": {
                             "Ratio_Selected": (s.ratio_sizes,{'default': default_ratio, "tooltip":'SDXL Native resolution selection'}),
                             "Width_Override": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip":'Overrides Width'}),
                             "Height_Override": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip":'Overrides Height'}),
                             "Batch_Count": ("INT", {"default": 1, "min": 1, "max": 1125899906842624, "tooltip":'Number of seeded batches'}),
                             "Batch_Size": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip":'Number of batched sub-images'}),
                             "First_Seed":("INT:seed", {"default": 1, "min": 1, "max": 1125899906842624, "tooltip":'Initial seed'}),
                             "Batch_Seeds": ("STRING", {"multiline": True, "default": "", "tooltip":'Seed Override for easy chaining. Can deal with hyphen separation (1234-1235) and line breaks.'})},
                "optional": {"Reseed_Latents": ("LATENT", {"tooltip":'Input latents to be rebatched and reseeded with current seed options'})}}

    RETURN_TYPES = ('LATENT','INT', 'STRING', 'INT', 'INT')
    RETURN_NAMES = ('Latents','Seeds', 'Seed String', 'Width', 'Height')
    OUTPUT_IS_LIST = (True, True, False, False, False )

    OUTPUT_TOOLTIPS = ('Latents',
                      'Batch Seeds',
                      'Seed String for easy chaining',
                      'Latent Height',
                      'Latent Width',)

    FUNCTION = 'gimmeLatent'
    CATEGORY = "Hellrunner's"
    DESCRIPTION = "Latent seed and batch controller with extra information outputs, so it can be used as resolution and seed master."

    @staticmethod
    def get_batch(latents, list_ind, offset):
        '''prepare a batch out of the list of latents'''
        samples = latents[list_ind]['samples']
        shape = samples.shape
        mask = latents[list_ind]['noise_mask'] if 'noise_mask' in latents[list_ind] else torch.ones((shape[0], 1, shape[2]*8, shape[3]*8), device='cpu')
        if mask.shape[-1] != shape[-1] * 8 or mask.shape[-2] != shape[-2]:
            torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[-2]*8, shape[-1]*8), mode="bilinear")
        if mask.shape[0] < samples.shape[0]:
            mask = mask.repeat((shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
        if 'batch_index' in latents[list_ind]:
            batch_inds = latents[list_ind]['batch_index']
        else:
            batch_inds = [x+offset for x in range(shape[0])]
        return samples, mask, batch_inds

    @staticmethod
    def get_slices(indexable, num, batch_size):
        '''divides an indexable object into num slices of length batch_size, and a remainder'''
        slices = []
        for i in range(num):
            slices.append(indexable[i*batch_size:(i+1)*batch_size])
        if num * batch_size < len(indexable):
            return slices, indexable[num * batch_size:]
        else:
            return slices, None
    
    @staticmethod
    def slice_batch(self, batch, num, batch_size):
        result = [self.get_slices(x, num, batch_size) for x in batch]
        return list(zip(*result))

    @staticmethod
    def cat_batch(batch1, batch2):
        if batch1[0] is None:
            return batch2
        result = [torch.cat((b1, b2)) if torch.is_tensor(b1) else b1 + b2 for b1, b2 in zip(batch1, batch2)]
        return result

    def rebatch(self, latents, batch_size):

        output_list = []
        current_batch = (None, None, None)
        processed = 0

        for i in range(len(latents)):
            next_batch = self.get_batch(latents, i, processed)
            processed += len(next_batch[2])
            if current_batch[0] is None:
                current_batch = next_batch
            elif next_batch[0].shape[-1] != current_batch[0].shape[-1] or next_batch[0].shape[-2] != current_batch[0].shape[-2]:
                sliced, _ = self.slice_batch(self, current_batch, 1, batch_size)
                output_list.append({'samples': sliced[0][0], 'noise_mask': sliced[1][0], 'batch_index': sliced[2][0]})
                current_batch = next_batch
            else:
                current_batch = self.cat_batch(current_batch, next_batch)

            if current_batch[0].shape[0] > batch_size:
                num = current_batch[0].shape[0] // batch_size
                sliced, remainder = self.slice_batch(self, current_batch, num, batch_size)
                
                for i in range(num):
                    output_list.append({'samples': sliced[0][i], 'noise_mask': sliced[1][i], 'batch_index': sliced[2][i]})

                current_batch = remainder

        if current_batch[0] is not None:
            sliced, _ = self.slice_batch(self, current_batch, 1, batch_size)
            output_list.append({'samples': sliced[0][0], 'noise_mask': sliced[1][0], 'batch_index': sliced[2][0]})

        for s in output_list:
            if s['noise_mask'].mean() == 1.0:
                del s['noise_mask']

        return (output_list,)


    def gimmeLatent(self, First_Seed, Ratio_Selected, Width_Override = 0, Height_Override = 0, Batch_Count = 1, Batch_Size = 1, Batch_Seeds = "", Reseed_Latents=None):

        outLatents = []
        outSeeds = []
        outSeedString = ""
        makeLatents = True

        width = Width_Override
        if Width_Override <= 0:
            width = int(self.ratio_dict[Ratio_Selected]["width"])

        height = Height_Override
        if Height_Override <= 0:
            height = int(self.ratio_dict[Ratio_Selected]["height"])

        linebreaks = Batch_Seeds.split('\n')
        Seedlist = []

        for linebreak in linebreaks:
            lines = linebreak.split('-')
            for line in lines:
                stripLine = line.strip()
                if stripLine.isdigit():
                    Seedlist.append(stripLine)

        if (Reseed_Latents is not None and len(Reseed_Latents)>0):
            width = int(Reseed_Latents["samples"].shape[3] * 8)
            height = int(Reseed_Latents["samples"].shape[2] * 8)
            outLatents = self.rebatch([Reseed_Latents], Batch_Size)[0]
            Batch_Count = len(outLatents)
            makeLatents = False

        if len(Seedlist) == 0:
            Seedlist.append(First_Seed)
        
        for i in range(Batch_Count):
            if makeLatents:
                latent = torch.zeros([Batch_Size, 4, height // 8, width // 8])
                outLatents.append({"samples":latent})

            if i != 0:
                outSeedString += "-"

            newSeed = 0
            if i < len(Seedlist):
                newSeed = Seedlist[i]
            else:
                newSeed = random.randint(1, 1125899906842624)

            outSeeds.append(newSeed)
            outSeedString+=str(newSeed)
        return (outLatents, outSeeds, outSeedString, width, height)


NODE_CLASS_MAPPINGS = {
    "MagicalSaveNode": MagicalSaveNode, 
    "ThermalLatenator": thermalLatenator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicalSaveNode": "Magical Save Node", 
    "ThermalLatenator": "Thermal Latenator",
}