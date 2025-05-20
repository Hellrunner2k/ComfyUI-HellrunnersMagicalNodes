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
import hashlib

import nodes
import node_helpers
import comfy.utils

class MagicalSaveNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {"images": ("IMAGE", ),
                    "Active": ("BOOLEAN", {"default": True, "label_on":"On", "label_off":"Off", "tooltip":'Boolean On/Off Switch for better integration in complex comfy-flows'}),
                    "Output_Path": ("STRING", {"default": '[time(%Y-%m-%d)]', "multiline": False, "tooltip":'Subfolder Path into "output"'}),
                    "Name": ("STRING", {"default": "ComfyUI", "tooltip":'File Name'}),
                    "Stamp": ("STRING", {"default": '%H-%M-%S-%f', "multiline": False, "tooltip":'timestamp format see strftime()'}),
                    "Extension": (['png', 'jpg', 'tiff', 'bmp', 'none'],{"default":'png', "tooltip":'Image Type'}),
                    "Quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip":'jpg compression 1-100, png compression 0-9  (if > 9 = 0 lossless)'}),
                    "Save_gen_data_to_txt": ("BOOLEAN", {"default": True, "label_on":"On", "label_off":"Off", "tooltip":'On saves meta-data based on renamed nodes (right-click -> "Title") and the comfy-flow to a text file'}),
                    "Save_gen_data_to_png": ("BOOLEAN", {"default": False, "label_on":"On", "label_off":"Off", "tooltip":'On saves meta-data based on renamed nodes (right-click -> "Title") and the comfy-flow to a png image'}),
                    "Formatting": (["Human Readable"],{"default":"Human Readable", "tooltip":'Meta Data Format. Included for future expandability without node breakage'}),
                },
               "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                
               }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True

    CATEGORY = "Hellrunner's"
    DESCRIPTION = 'Compiles meta-data based on renamed nodes (right-click -> "Title") and optionally includes it and the comfy-flow in a text file and/or a png image.'

    def save_images(self, images, Active, Output_Path='[time(%Y-%m-%d)]', Name='ComfyUI',Stamp='%H-%M-%S-%f', Extension='png', Quality=95, Save_gen_data_to_txt=True, Save_gen_data_to_png=False, Formatting='Human Readable', prompt=None, extra_pnginfo=None):
        
        def replace_custom_time(match):
            format_code = match.group(1)
            return time.strftime(format_code, time.localtime(time.time()))

        def writeTextFile(file, content):
            try:
                with open(file, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(content)
            except OSError:
                print(str(f"Unable to save file `{file}`"))

        if not Active:
            return ()

        tokens = {'[time]': str(time.time())}

        tokens['[time]'] = str(time.time())
        if '.' in tokens['[time]']:
            tokens['[time]'] = tokens['[time]'].split('.')[0]

        for token, value in tokens.items():
            if token.startswith('[time('):
                continue
            Output_Path = Output_Path.replace(token, value)

        path = re.sub(r'\[time\((.*?)\)\]', replace_custom_time, Output_Path)

        results = list()

        if Name == "":
            Name="ComfyUI"
        addPth = Name.replace('\ ','/').split("/")
        filename = addPth.pop()

        if Stamp == "":
            Stamp == '%H-%M-%S-%f'

        stamp = datetime.datetime.now().strftime(Stamp)
        filename = f'{filename}-{stamp}'

        for folder in addPth:
            path = os.path.join(path, folder)

        full_output_folder = path

        if not os.path.isdir(path):
            full_output_folder = os.path.join(folder_paths.get_output_directory(), path)
        

        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder, exist_ok=True)

        fn=f'{filename}'
        file = f'{fn}.{Extension}'
        counter = 0

        while os.path.exists(os.path.join(full_output_folder, file)):
            counter += 1
            fn=f"{filename}_{counter:05}"
            file = f"{fn}.{Extension}"

        full_path = os.path.join(full_output_folder, file)

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

            if Save_gen_data_to_txt:
                writeTextFile(os.path.join(full_output_folder, f"{filename}_{counter:05}.txt"), txtMeta)

            if not Save_gen_data_to_png:
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
            fn=f"{filename}_{counter:05}"
            file = f"{fn}.{Extension}"
        
            while os.path.exists(os.path.join(full_output_folder, file)):
                counter += 1
                fn=f"{filename}_{counter:05}"
                file = f"{fn}.{Extension}"

        return { "ui": { "images": results } }

DEF_REZ_DICT = {
        "1:1 1024x1024 1MP #": {"width": 1024, "height":  1024},
        "8:5 1216x768 1MP [_]": {"width": 1216, "height":  768},
        "4:3 1152x896 1MP [_]": {"width": 1152, "height":  896},
        "3:2 1216x832 1MP [_]": {"width": 1216, "height":  832},
        "7:5 1176x840 1MP [_]": {"width": 1176, "height":  840},
        "16:9 1344x768 1MP [_]": {"width": 1344, "height":  768},
        "21:9 1536x640 1MP [_]": {"width": 1536, "height":  640},
        "19:9 1472x704 1MP [_]": {"width": 1472, "height":  704},
        "3:4 896x1152 1MP []": {"width": 896, "height":  1152},
        "2:3 832x1216 1MP []": {"width": 832, "height":  1216},
        "5:7 840x1176 1MP []": {"width": 840, "height":  1176},
        "9:16 768x1344 1MP []": {"width": 768, "height":  1344},
        "9:21 640x1536 1MP []": {"width": 640, "height":  1536},
        "5:8 768x1216 1MP []": {"width": 768, "height":  1216},
        "9:19 704x1472 1MP []": {"width": 704, "height":  1472}
        }

class thermalLatenator:

    @classmethod
    def INPUT_TYPES(s):
        s.ratio_dict = DEF_REZ_DICT
        s.ratio_sizes = list(s.ratio_dict.keys())
        default_ratio = s.ratio_sizes[0]

        return {"required": {
                             "Ratio_Selected": (s.ratio_sizes,{'default': default_ratio, "tooltip":'SDXL Native resolution selection'}),
                             "Width_Override": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip":'>0 Overrides Width'}),
                             "Height_Override": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip":'>0 Overrides Height'}),
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
                      'Latent Width',
                      'Latent Height',)

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

        return(output_list,)


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
        return(outLatents, outSeeds, outSeedString, width, height)

class LoadMaskMap:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and (f.endswith(".bmp") or f.endswith(".BMP"))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True, "tooltip":'RGB Mask-Map as bmp'})}, 
                    "optional": {
                        "Width": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip":'Width Override, >0 initializes scaling'}),
                        "Height": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip":'Height Override, >0 initializes scaling'}),
                        "upscale_method": (s.upscale_methods,{"default": "lanczos", "tooltip":'Upscaling method to use if scaling'})}}


    RETURN_TYPES = ("MASKMAP","MASK","MASK","MASK")
    RETURN_NAMES = ('Mask-Map','Red (Center Piece)','Green (Theme)', 'Blue (Background)')
    OUTPUT_TOOLTIPS = ('Mask-Map',
                       'Red Mask (Center Piece)',
                       'Green Mask (Theme)',
                       'Blue Mask (Background)')

    FUNCTION = 'maskIt'
    CATEGORY = "Hellrunner's/Mask-Maps"
    DESCRIPTION = "Open, optionally scale and split a Mask-Map bmp. Smooth gradient masks with 100% combined prompt coverage. Ready for use in one go."

    def maskIt(self, image, Width, Height, upscale_method):
        image_path = folder_paths.get_annotated_filepath(image)
        i = node_helpers.pillow(Image.open, image_path)
        i = node_helpers.pillow(ImageOps.exif_transpose, i)
        if i.getbands() != ("R", "G", "B"):
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            i = i.convert("RGB")

        mask = {}
        #c = channel[0].upper()
        for c in i.getbands():
            mask[c] = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            mask[c] = torch.from_numpy(mask[c]).unsqueeze(0)
            if Width > 0 or Height > 0:
                samples = mask[c].movedim(-1,1)

                if Width == 0:
                    Width = max(1, round(samples.shape[3] * Height / samples.shape[2]))
                elif Height == 0:
                    Height = max(1, round(samples.shape[2] * Width / samples.shape[3]))

                mask[c] = comfy.utils.common_upscale(samples, Width, Height, upscale_method, "disabled")
                mask[c] = mask[c].movedim(1,-1)
        return (mask, mask['R'], mask['G'], mask['B'],)

    @classmethod
    def IS_CHANGED(s, image, Width, Height, upscale_method):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

class MaskMapPromptMix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            },"optional":{
            "Base": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip":'Base Prompt strength'}),
            "Red": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip":'Red Prompt strength'}),
            "Green": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip":'Green Prompt strength'}),
            "Blue": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip":'Blue Prompt strength'}),
            "Order": ("STRING", {"default":"Base,Red,Green,Blue", "multiline": False, "dynamicPrompts": False, "tooltip":'Prompt Sequence. Comma separated sequence of outputs (Base,Red,Green,Blue)'}),
            }}
    RETURN_TYPES = ("MMPMIX",)
    RETURN_NAMES = ('MMPMix',)
    OUTPUT_TOOLTIPS = ('Mask-Map Mix',)
    FUNCTION = "encode"

    CATEGORY = "Hellrunner's/Mask-Maps"
    DESCRIPTION = "Settings container for the Mask-Map Prompt."

    def encode(self, Base, Red, Green,Blue,Order):

        out={"Base":Base,"Red":Red,"Green":Green,"Blue":Blue,"Order":Order,}

        return (out, )

class MaskMapPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            },"optional":{
            "Base_Prompt": ("STRING", {"default":"", "multiline": True, "dynamicPrompts": True, "tooltip":'Base Prompt'}),
            "Base_Prompt_MMPMix": ("MMPMIX", {"default":{}, "tooltip":'Base Prompt MMPMix value package',}),
            "Base_Negative": ("STRING", {"default":"", "multiline": False, "dynamicPrompts": True, "tooltip":'Base Negative Prompt',}),
            "Base_Negative_MMPMix": ("MMPMIX", {"default":{}, "tooltip":'Base Negative Prompt MMPMix value package',}),

            "Red_Prompt": ("STRING", {"default":"", "multiline": True, "dynamicPrompts": True, "tooltip":'Red Prompt',}),
            "Red_Prompt_MMPMix": ("MMPMIX", {"default":{}, "tooltip":'Red Prompt MMPMix value package',}),
            "Red_Negative": ("STRING", {"default":"", "multiline": False, "dynamicPrompts": True, "tooltip":'Red Negative Prompt',}),
            "Red_Negative_MMPMix": ("MMPMIX", {"default":{}, "tooltip":'Red Negative Prompt MMPMix value package',}),

            "Green_Prompt": ("STRING", {"default":"", "multiline": True, "dynamicPrompts": True, "tooltip":'Green Prompt',}),
            "Green_Prompt_MMPMix": ("MMPMIX", {"default":{}, "tooltip":'Green Prompt MMPMix value package',}),
            "Green_Negative": ("STRING", {"default":"", "multiline": False, "dynamicPrompts": True, "tooltip":'Green Negative Prompt',}),
            "Green_Negative_MMPMix": ("MMPMIX", {"default":{}, "tooltip":'Green Negative Prompt MMPMix value package',}),

            "Blue_Prompt": ("STRING", {"default":"", "multiline": True, "dynamicPrompts": True, "tooltip":'Blue Prompt',}),
            "Blue_Prompt_MMPMix": ("MMPMIX", {"default":{}, "tooltip":'Blue Prompt MMPMix value package',}),
            "Blue_Negative": ("STRING", {"default":"", "multiline": False, "dynamicPrompts": True, "tooltip":'Blue Negative Prompt',}),
            "Blue_Negative_MMPMix": ("MMPMIX", {"default":{}, "tooltip":'Blue Negative Prompt MMPMix value package',}),
            }}
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING","STRING",)
    RETURN_NAMES = ('Base_Prompt','Base_Negative','Red_Prompt','Red_Negative','Green_Prompt','Green_Negative','Blue_Prompt','Blue_Negative','Combined_Prompt','Combined_Negative',)
    OUTPUT_TOOLTIPS = ('Base Prompt','Base Negative Prompt','Red Prompt','Red Negative Prompt','Green Prompt','Green Negative Prompt','Blue Prompt','Blue Negative Prompt','Combined Prompt (wip ignore)','Combined Negative Prompt (wip ignore)',)
    FUNCTION = "encode"

    CATEGORY = "Hellrunner's/Mask-Maps"
    DESCRIPTION = "Mix and merge weighted prompts for Mask-Maps"



    def domagig(self,out,typ,PoN,args):
        defuhld = {"Base":1.0,"Red":1.0,"Green":1.0,"Blue":1.0,"Order":"Base,Red,Green,Blue",}
        ty = ["Base","Red","Green","Blue"]
        args["cPromptMix"] = {"Base":[],"Red":[],"Green":[],"Blue":[],"Order":[[],[],[],[]],}
        args["cNegativeMix"] = {"Base":[],"Red":[],"Green":[],"Blue":[],"Order":[[],[],[],[]],}
        
        n = f'{typ}_{PoN}'
        out[n]= ""
        mixNm = f'c{PoN}Mix'

        if typ == "Combined":

            return out

        
        nMix = f'{n}_MMPMix'

        mix = defuhld
        if nMix in args.keys() and not args[nMix]==None:
            mix = args[nMix]

        order = mix["Order"].split(",")
        order = order[:4]
        cou=0
        for k in order:
            if k in ty:
                ordN = f'{k}_{PoN}'
                out[n] += f'{"" if cou==0 else ","}({args[ordN]}:{round(mix[typ], 3)})'
                args[mixNm][f'{k}'].append(mix[typ])
                args[mixNm]['Order'][cou].append(f'{k}')
                cou+=1
        print(f'{args[mixNm]}')

        return out

    def encode(self,
               Base_Prompt,Base_Negative,Red_Prompt,Red_Negative,
               Green_Prompt,Green_Negative,Blue_Prompt,Blue_Negative,

               Base_Prompt_MMPMix=None,Base_Negative_MMPMix=None,Red_Prompt_MMPMix=None,Red_Negative_MMPMix=None,
               Green_Prompt_MMPMix=None,Green_Negative_MMPMix=None,Blue_Prompt_MMPMix=None,Blue_Negative_MMPMix=None
                
              ):

        args={"Base_Prompt":Base_Prompt,"Base_Prompt_MMPMix":Base_Prompt_MMPMix,"Base_Negative":Base_Negative,"Base_Negative_MMPMix":Base_Negative_MMPMix,
              "Red_Prompt":Red_Prompt,"Red_Prompt_MMPMix":Red_Prompt_MMPMix,"Red_Negative":Red_Negative,"Red_Negative_MMPMix":Red_Negative_MMPMix,
              "Green_Prompt":Green_Prompt,"Green_Prompt_MMPMix":Green_Prompt_MMPMix,"Green_Negative":Green_Negative,"Green_Negative_MMPMix":Green_Negative_MMPMix,
              "Blue_Prompt":Blue_Prompt,"Blue_Prompt_MMPMix":Blue_Prompt_MMPMix,"Blue_Negative":Blue_Negative,"Blue_Negative_MMPMix":Blue_Negative_MMPMix,
              }

        typ = ["Base","Red","Green","Blue","Combined"]
      
        
        out={}

        for ty in typ:
            out = self.domagig(out, ty, "Prompt", args)
            out = self.domagig(out, ty, "Negative", args)


        return (out["Base_Prompt"],out["Base_Negative"], out["Red_Prompt"],out["Red_Negative"],out["Green_Prompt"],out["Green_Negative"],
               out["Blue_Prompt"],out["Blue_Negative"],out["Combined_Prompt"],out["Combined_Negative"],)

class BufferedEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "Active": ("BOOLEAN", {"default": True, "label_on":"On", "label_off":"Off", "tooltip":'Boolean On/Off Switch for better integration in complex flows'}),
                        "Output_Path": ("STRING", {"default": '[time(%Y-%m-%d)]', "multiline": False, "tooltip":'Subfolder Path into "output"'}),
                        "Name": ("STRING", {"default": "Buffer", "tooltip":'Buffer File Name (Name.conditioning)'}),
                        "Override": ("BOOLEAN", {"default": False, "label_on":"On", "label_off":"Off", "tooltip":'Force override the buffer'}),
                        "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                        "Encoder_1": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "The Encoder_1 used for encoding the text."}),
                        "use_Encoder_2": ("BOOLEAN", {"default": False, "label_on":"On", "label_off":"Off", "tooltip":'Enable second encoder'}),
                        "Encoder_2": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "The Encoder_2 used for encoding the text."}),
                        "use_Encoder_3": ("BOOLEAN", {"default": False, "label_on":"On", "label_off":"Off", "tooltip":'Enable 3rd encoder'}),
                        "Encoder_3": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "The Encoder_3 used for encoding the text."}),
                        "use_Encoder_4": ("BOOLEAN", {"default": False, "label_on":"On", "label_off":"Off", "tooltip":'Enable 4th encoder'}),
                        "Encoder_4": (folder_paths.get_filename_list("text_encoders"), {"tooltip": "The Encoder_4 used for encoding the text."}),
                        "clip_skip": ("INT", {"default": -2, "min": -128, "max": -1, "tooltip": "Last Layer"}),
                        "model_type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace"],{"tooltip": "Model Type to load encoders for."} ),
                        "load_device": (["default", "cpu"], {"advanced": True}),
                        },
                "optional":{
                        "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                        "LoRABox": ("LORABOX", {"default": None, "tooltip":'LoRABox'}),
                        "model": ("MODEL", { "tooltip": "The model to apply the LoRABox to"}),
                }}
                            
    RETURN_TYPES = ("CONDITIONING","CLIP","MODEL",)
    RETURN_NAMES = ('CONDITIONING','CLIP or not','MODEL or not',)
    OUTPUT_TOOLTIPS = ("Conditioning","Used Clip, if encode happens. Used for chaining encoders so loading does not happen multiple times","Used Model, if LoRABox is not empty",)
    FUNCTION = 'energize'

    CATEGORY = "Hellrunner's/Utils"
    DESCRIPTION = "Encodes and buffers a conditioning on the hard drive. Uses that buffer if enabled."

    def energize(self, Active=True, Output_Path = "[time(%Y-%m-%d)]", Name="Snip", Override=False, text="", 
                 clip=None, Encoder_1="",use_Encoder_2=False,Encoder_2="",use_Encoder_3=False,Encoder_3="",use_Encoder_4=False,Encoder_4="",
                 clip_skip=-2, model_type="stable_diffusion",load_device="default", LoRABox=None, model=None,): 

        def replace_custom_time(match):
            format_code = match.group(1)
            return time.strftime(format_code, time.localtime(time.time()))

        def LoRAapply(lb, inClip,inModel):
            if (inClip == None and inModel == None) or lb==None or len(lb)==0:
                return (inClip,inModel)

            for LoRA in lb:
                if LoRA["Active"]:
                    lora_path = folder_paths.get_full_path("loras", LoRA["Name"])
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            
                    inModel, inClip = comfy.sd.load_lora_for_models(inModel, inClip, lora, LoRA["Model"], LoRA["TE"])

            return (inClip,inModel)


        def encode(clip, text, LoBo,Mod):
            if clip is None:
                clipery = []
                clipery.append(folder_paths.get_full_path_or_raise("text_encoders", Encoder_1))
                if use_Encoder_2:
                    clipery.append(folder_paths.get_full_path_or_raise("text_encoders", Encoder_2))
                if use_Encoder_3:
                    clipery.append(folder_paths.get_full_path_or_raise("text_encoders", Encoder_3))
                if use_Encoder_4:
                    clipery.append(folder_paths.get_full_path_or_raise("text_encoders", Encoder_4))
                
                if len(clipery) == 0:
                    raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
                clip_type = getattr(comfy.sd.CLIPType, model_type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

                model_options = {}
                if load_device == "cpu":
                    model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
                clip = comfy.sd.load_clip(ckpt_paths=clipery, embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
                clip.clip_layer(clip_skip)
  
            clip,Mod = LoRAapply(LoBo,clip,Mod)
            tokens = clip.tokenize(text)
            return (clip.encode_from_tokens_scheduled(tokens), clip, Mod)

        def saveCond(Cond, Pth,folderPth, folderEx):
            if not folderEx:
                os.makedirs(folderPth, exist_ok=True)

            Buffer = {}
            print(f'{Cond}')
            Buffer["Cond"] = Cond[0][0]
            ExtraVars={}
            if len(Cond[0])>1:
                for val in Cond[0][1].keys():
                    #print(f'{type(Cond[0][1][val])}')
                    ExtraVars[val] = Cond[0][1][val]
                    
                #if "mask" in Cond[0][1]:
                    #Buffer["Msk"] = Cond[0][1]["mask"]
            Buffer["ExtraVars"] = torch.frombuffer(bytearray(json.dumps(ExtraVars), 'utf-8'), dtype=torch.uint8)
            try:
                save_file(Buffer, Pth)
            except OSError:
                print(str(f"Unable to save file `{Pth}`"))

        def loadCond(Pth,):
            loaded=None
            try:
                file = open(Pth, "rb")
                data = file.read()
                loaded = load_file(data)
                file.close()
            except OSError:
                print(str(f"Unable to load file `{Pth}`"))
            if not loaded==None:
                ExtraVars = json.loads("".join(map(chr, loaded["ExtraVars"])))
                print(f'{ExtraVars}')
                return([[loaded["Cond"],ExtraVars]])
            return

        tokens = {'[time]': str(time.time())}

        tokens['[time]'] = str(time.time())
        if '.' in tokens['[time]']:
            tokens['[time]'] = tokens['[time]'].split('.')[0]

        for token, value in tokens.items():
            if token.startswith('[time('):
                continue
            Output_Path = Output_Path.replace(token, value)

        path = re.sub(r'\[time\((.*?)\)\]', replace_custom_time, Output_Path)

        if Name == "":
            Name="Buffer"
        addPth = Name.replace('\ ','/').split("/")
        filename = addPth.pop() 

        for folder in addPth:
            path = os.path.join(path, folder)

        full_output_folder = path

        if not os.path.isdir(path):
            full_output_folder = os.path.join(folder_paths.get_output_directory(), path)
        
        file = f'{filename}.conditioning'
        full_path = os.path.join(full_output_folder, file)

        FileExist = os.path.exists(full_path)
        PathExist = False
        if FileExist:
            PathExist = True
        else:
            PathExist = os.path.exists(full_output_folder)

        if not Active:
            c, c2,model = encode(clip,text,LoRABox,model)
            if Override:
                saveCond(c,full_path,full_output_folder,PathExist,)
            return (c, c2,model,)
        else:
            if Override:
                c, c2,model = encode(clip,text,LoRABox,model)
                saveCond(c,full_path,full_output_folder,PathExist,)
                return (c, c2,model,)
            else:
                if FileExist:
                    clip,model = LoRAapply(LoRABox,clip,model)
                    return(loadCond(full_path), clip,model,)
                else:
                    c, c2,model = encode(clip,text,LoRABox,model)
                    saveCond(c,full_path,full_output_folder,PathExist,)
                    return (c, c2,model,)
            
        return ()

class TEAce:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            },"optional":{
            "prompt": ("STRING", {"default":"", "multiline": True, "dynamicPrompts": True, "tooltip":'Detailed description of the song.'}),
            "lyrics": ("STRING", {"default":"", "multiline": True, "dynamicPrompts": True, "tooltip":'Lyrics - it also responds to short instruction in []. [chorus] [verse] [instrumental]'}),
            "prompt_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip":'substancial cfg multiplier'}),
            "lyrics_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip":'structural cfg multiplier'}),
            "speaker_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip":'embedding cfg multiplier'}),
            "mask": ("MASK", {"tooltip":'Mask stretched over the entire track'}),
            "set_area_to_bounds": ("BOOLEAN", {"default": False, "label_on":"On", "label_off":"Off", "tooltip":'restrict attention to mask bounds'}),
            "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip":'Opacity of the masked area'}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "Hellrunner's/Utils"
    DESCRIPTION = "Prepare ACE-Step Conditionig"

    def encode(self, clip, prompt="", lyrics="",prompt_strength=1.0, lyrics_strength=1.0, speaker_strength=1.0, mask=None, set_area_to_bounds=False,mask_strength=1.0):

        tokens = clip.tokenize(prompt, lyrics=lyrics)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if not mask == None:
            if len(mask.shape) < 3:
                mask = mask.unsqueeze(0)

            conditioning = node_helpers.conditioning_set_values(conditioning, {"mask": mask,
                                                                    "set_area_to_bounds": set_area_to_bounds,
                                                                    "mask_strength": mask_strength})

        conditioning = node_helpers.conditioning_set_values(conditioning, {"prompt_strength": prompt_strength})
        conditioning = node_helpers.conditioning_set_values(conditioning, {"lyrics_strength": lyrics_strength})
        conditioning = node_helpers.conditioning_set_values(conditioning, {"speaker_strength": speaker_strength})

        return (conditioning, )

# needs hax ... in comfy\model_base.py ---------------------------------------------------------------


# class ACEStep(BaseModel):
#     def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
#         super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.ace.model.ACEStepTransformer2DModel)
#
#     def extra_conds(self, **kwargs):
#         out = super().extra_conds(**kwargs)
#         noise = kwargs.get("noise", None)
#
#         cross_attn = kwargs.get("cross_attn", None)
#         if cross_attn is not None:
#             out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
#
#         conditioning_lyrics = kwargs.get("conditioning_lyrics", None)
#         if cross_attn is not None:
#             out['lyric_token_idx'] = comfy.conds.CONDRegular(conditioning_lyrics)
#         out['speaker_embeds'] = comfy.conds.CONDRegular(torch.zeros(noise.shape[0], 512, device=noise.device, dtype=noise.dtype))
#       
#         out['prompt_strength'] = comfy.conds.CONDConstant(kwargs.get("prompt_strength", 1.0))         <----------
#         out['lyrics_strength'] = comfy.conds.CONDConstant(kwargs.get("lyrics_strength", 1.0))
#         out['speaker_strength'] = comfy.conds.CONDConstant(kwargs.get("speaker_strength", 1.0))       <----------
#         return out

# ..... in comfy\ldm\ace\model.py ... in encode  at around 290 -------------------------------------------------------------------------

#         # genre embedding
#         encoder_text_hidden_states = self.genre_embedder(encoder_text_hidden_states)

#         # lyric
#         encoder_lyric_hidden_states = self.forward_lyric_encoder(
#             lyric_token_idx=lyric_token_idx,
#             lyric_mask=lyric_mask,
#             out_dtype=encoder_text_hidden_states.dtype,
#         )

#         encoder_text_hidden_states *= prompt_strength         <----------
#         encoder_lyric_hidden_states *= lyrics_strength
#         encoder_spk_hidden_states *= speaker_strength         <----------

#         encoder_hidden_states = torch.cat([encoder_spk_hidden_states, encoder_text_hidden_states, encoder_lyric_hidden_states], dim=1)

#         encoder_hidden_mask = None


# ------ AND on the function inputs ... encode -----------------------------------------------------


#    def encode(
#         self,
#         encoder_text_hidden_states: Optional[torch.Tensor] = None,
#         text_attention_mask: Optional[torch.LongTensor] = None,
#         speaker_embeds: Optional[torch.FloatTensor] = None,
#         lyric_token_idx: Optional[torch.LongTensor] = None,
#         lyric_mask: Optional[torch.LongTensor] = None,
#         prompt_strength=1.0,      <----------
#         lyrics_strength=1.0,
#         speaker_strength=1.0,     <----------
#     ):

#         bs = encoder_text_hidden_states.shape[0]
#         device = encoder_text_hidden_states.device


# ------ AND on the function inputs ... forward -------------------------------------------------------


#    def forward(
#         self,
#         x,
#         timestep,
#         attention_mask=None,
#         context: Optional[torch.Tensor] = None,
#         text_attention_mask: Optional[torch.LongTensor] = None,
#         speaker_embeds: Optional[torch.FloatTensor] = None,
#         lyric_token_idx: Optional[torch.LongTensor] = None,
#         lyric_mask: Optional[torch.LongTensor] = None,
#         block_controlnet_hidden_states: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
#         controlnet_scale: Union[float, torch.Tensor] = 1.0,
#         prompt_strength=1.0,      <----------
#         lyrics_strength=1.0,
#         speaker_strength=1.0,     <----------
#         **kwargs
#     ):
#         hidden_states = x
#         encoder_text_hidden_states = context
#         encoder_hidden_states, encoder_hidden_mask = self.encode(
#             encoder_text_hidden_states=encoder_text_hidden_states,
#             text_attention_mask=text_attention_mask,
#             speaker_embeds=speaker_embeds,
#             lyric_token_idx=lyric_token_idx,
#             lyric_mask=lyric_mask,
#             prompt_strength=prompt_strength,      <----------
#             lyrics_strength=lyrics_strength,
#             speaker_strength=speaker_strength,    <----------
#         )

#         output_length = hidden_states.shape[-1]

#         output = self.decode(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_hidden_mask=encoder_hidden_mask,
#             timestep=timestep,
#             output_length=output_length,
#             block_controlnet_hidden_states=block_controlnet_hidden_states,
#             controlnet_scale=controlnet_scale,
#         )

#         return output

NODE_CLASS_MAPPINGS = {
    "MagicalSaveNode": MagicalSaveNode, 
    "ThermalLatenator": thermalLatenator,
    "LoadMaskMap": LoadMaskMap,
    "MaskMapPrompt": MaskMapPrompt,
    "MaskMapPromptMix": MaskMapPromptMix,
    "BufferedEncoder": BufferedEncoder,
    "TEAce": TEAce,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicalSaveNode": "Magical Save Node", 
    "ThermalLatenator": "Thermal Latenator",
    "LoadMaskMap": "Mask-Map Loader (HMN)",
    "MaskMapPrompt": "Mask-Map Prompt (HMN)",
    "MaskMapPromptMix": "Mask-Map Prompt Mix (HMN)",
    "BufferedEncoder": "Magical Encoder (HMN)",
    "TEAce": "Encode Ace (HMN)",
}