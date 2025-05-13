import torch
import os
import json
import time
import re

import folder_paths

import nodes
import node_helpers
import numpy as np

from safetensors.torch import save_file, load

def getVer(full_path):
    tmp = 0
    file = open(full_path, "rb")
    data = file.read()
    loaded = load(data)
    tmp = int(loaded["Ver"][0][0])
    file.close()
    return tmp

def getInfo(full_path):
    tmp = 0
    file = open(full_path, "rb")
    data = file.read()
    loaded = load(data)
    ver = int(loaded["Ver"][0][0])
    info = json.loads("".join(map(chr, loaded["Info"])))
    file.close()
    return ver, info

class SaveMojo:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"Active": ("BOOLEAN", {"default": True, "label_on":"On", "label_off":"Off", "tooltip":'Boolean On/Off Switch for better integration in complex comfy-flows'}),
                             "Output_Path": ("STRING", {"default": 'Hellkars/Mojos', "multiline": False, "tooltip":'Subfolder Path into "output"'}),
                             "Name": ("STRING", {"default": "ShareMy", "tooltip":'Mojo File Name'}),
                             "Override": ("BOOLEAN", {"default": True, "label_on":"On", "label_off":"Off", "tooltip":'Saves over the exact file name on "True/On" or creates numbered unique files on "False/Off"'}),
                             "latent": ("LATENT", {"tooltip":'Latents to Mojo file. Provide decoding info.'}),
                             "Positive_Mojo": ("MOJO", {"tooltip":'Positive Mojo Flow'}), 
                             "Negative_Mojo": ("MOJO", {"tooltip":'Negative Mojo Flow'}),
                             "Info": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False, "tooltip":'Extra Mojo information. Provide models used. With what VAE to decode the latent... and such'})}}
                            
    OUTPUT_IS_LIST = ()
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = 'energize'
    CATEGORY = "Hellrunner's/Mojo"
    DESCRIPTION = "Saves a .mojo file using safetensors with lightweight information, to further refine or recreate an image. Provide various clip sources to make your mojo more versatile."

    def energize(self, Active, Output_Path='Hellkars/Mojos', Name='ShareMy', Override=True, latent=None, Positive_Mojo=None, Negative_Mojo=None, Info=""): 

        if not Active:
            return ()

        path = Output_Path

        if Name == "":
            Name="ShareMy"
        addPth = Name.replace('\ ','/').split("/")

        filename = addPth.pop()

        for folder in addPth:
            path = os.path.join(path, folder)

        full_output_folder = os.path.join(folder_paths.get_output_directory(), path)
        

        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder, exist_ok=True)

        file = f'{filename}.mojo'
        if not Override:

            List = os.listdir(full_output_folder)
            counter = 1
            for FileName in List:
                if (FileName.split(".")).pop() == 'mojo':
                    counter+=1
        
            file = f"{filename}_{counter:05}.mojo"
            while os.path.exists(os.path.join(full_output_folder, file)):
                counter += 1
                file = f"{filename}_{counter:05}.mojo"

        Buffer = {}
        TokIndex = 0
        Buffer["Lnt"] = latent["samples"]    

        for Tok in Positive_Mojo:
            TikIndex = 0
            for Tik in Tok[0]:
                for TE in Tik:
                    Buffer[f"Pos.{TokIndex}.Tik.{TikIndex}.{TE}"] = torch.frombuffer(bytearray(json.dumps(Tok[0][TikIndex][TE]), 'utf-8'), dtype=torch.uint8)
                TikIndex += 1

            for extra in Tok[1]:
                if type(Tok[1][extra]) is torch.Tensor:
                    Buffer[f"Pos.{TokIndex}.Mask.{extra}"] = Tok[1][extra].contiguous()
                else:
                    try:
                        Buffer[f"Pos.{TokIndex}.Mask.{extra}"] = torch.frombuffer(bytearray(json.dumps(Tok[1][extra]), 'utf-8'), dtype=torch.uint8)
                    except (TypeError):
                        print(f'{Tok[1][extra]} excluded')
            for extraToo in Tok[2]:
                if type(Tok[2][extraToo]) is torch.Tensor:
                    Buffer[f"Pos.{TokIndex}.Extra.{extraToo}"] = Tok[2][extraToo].contiguous()       
                else:
                    try:
                        Buffer[f"Pos.{TokIndex}.Extra.{extraToo}"] = torch.frombuffer(bytearray(json.dumps(Tok[2][extraToo]), 'utf-8'), dtype=torch.uint8)
                    except (TypeError):
                        print(f'{Tok[2][extraToo]} excluded')
            TokIndex += 1
        TokIndex = 0

        for Tok in Negative_Mojo:
            TikIndex = 0
            for Tik in Tok[0]:
                for TE in Tik:
                    Buffer[f"Neg.{TokIndex}.Tik.{TikIndex}.{TE}"] = torch.frombuffer(bytearray(json.dumps(Tok[0][TikIndex][TE]), 'utf-8'), dtype=torch.uint8)
                TikIndex += 1

            for extra in Tok[1]:
                if type(Tok[1][extra]) is torch.Tensor:
                    Buffer[f"Neg.{TokIndex}.Mask.{extra}"] = Tok[1][extra].contiguous()        
                else:
                    try:
                        Buffer[f"Neg.{TokIndex}.Mask.{extra}"] = torch.frombuffer(bytearray(json.dumps(Tok[1][extra]), 'utf-8'), dtype=torch.uint8)
                    except (TypeError):
                        print(f'{Tok[1][extra]} excluded')
            for extraToo in Tok[2]:
                if type(Tok[2][extraToo]) is torch.Tensor:
                    Buffer[f"Neg.{TokIndex}.Extra.{extraToo}"] = Tok[2][extraToo].contiguous()        
                else:
                    try:
                        Buffer[f"Neg.{TokIndex}.Extra.{extraToo}"] = torch.frombuffer(bytearray(json.dumps(Tok[2][extraToo]), 'utf-8'), dtype=torch.uint8)
                    except (TypeError):
                        print(f'{Tok[2][extraToo]} excluded')
            TokIndex += 1
        
        Buffer["Info"] = torch.frombuffer(bytearray(json.dumps(Info), 'utf-8'), dtype=torch.uint8)

        full_path = os.path.join(full_output_folder, file)
        if os.path.exists(full_path):
            Buffer["Ver"] = torch.tensor([[getVer(full_path)+1]]) 
            save_file(Buffer, full_path)
            print(f'{full_path} updated')
        else:
            Buffer["Ver"] = torch.tensor([[0]])
            save_file(Buffer, full_path)
            print(f'{full_path} saved')
        return ()

def makeMaskSlot(mask, strength, set_cond_area, Cond = None):
    set_area_to_bounds = False
    if set_cond_area != "default":
        set_area_to_bounds = True
    returnSlot = {"mask": mask, "set_area_to_bounds": set_area_to_bounds, "mask_strength": strength}
    C = []
    if Cond is not None:
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        maskSlot = {"mask": mask, "set_area_to_bounds": set_area_to_bounds, "mask_strength": strength}
        C = node_helpers.conditioning_set_values(Cond, maskSlot)
    return C, returnSlot

def condition(clip, Tokens, guidance, width, height, crop_w, crop_h, target_width, target_height, start_percent, end_percent,  mask, strength, set_cond_area):
    output = clip.encode_from_tokens(Tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")
    output["guidance"] = guidance
    output["width"] = width
    output["height"] = height
    output["crop_w"] = crop_w
    output["crop_h"] = crop_h
    output["target_width"] = target_width
    output["target_height"] = target_height

    C = [[cond, output]]

    C = node_helpers.conditioning_set_values(C, {"start_percent": start_percent,
                                                                "end_percent": end_percent})
    returnSlot = {}
    if mask is not None:
        C, returnSlot = makeMaskSlot(mask, strength, set_cond_area, C)

    return C, returnSlot

def ChToNr(choice):
    match choice:
        case "Main":
            return 0
        case "Additional":
            return 1
        case "Special":
            return 2
        case _:
            return 0

def NrToCh(number):
    match number:
        case 0:
            return "Main"
        case 1:
            return "Additional"
        case 2:
            return "Special"
        case _:
            return "Main"

def makeSettings(settings, guidance, width, height, crop_w, crop_h, target_width, target_height, start_percent, end_percent):
    settings["guidance"] = guidance
    settings["width"] = width
    settings["height"] = height
    settings["crop_w"] = crop_w
    settings["crop_h"] = crop_h
    settings["target_width"] = target_width
    settings["target_height"] = target_height
    settings["start_percent"] = 0.0
    settings["end_percent"]= 1.0

    if start_percent >=0.0:
        settings["start_percent"] = start_percent
    if end_percent >=0.0:
        settings["end_percent"] = end_percent
    
    return settings

def SetMojo(Mojo, guidance, width, height, crop_w, crop_h, target_width, target_height, start_percent=-1.0, end_percent=-1.0):

    for i in range(len(Mojo)):
        sp = start_percent
        ep = end_percent

        if sp < 0.0 and "start_percent" in Mojo[i][2]:
            sp = Mojo[i][2]["start_percent"]

        if ep < 0.0 and "end_percent" in Mojo[i][2]:
            ep = Mojo[i][2]["end_percent"]

        settings = makeSettings({}, guidance, width, height, crop_w, crop_h, target_width, target_height, sp, ep)

        Mojo[i][2] = settings


def ClipTest(clip):
    clipT = clip.tokenize("a woman")
    return list(clipT.keys())

def SqueezeMojo(Mojo_In, Squeeze_Mojo, Merge_Mojo=None, Strength=1.0):
    if Squeeze_Mojo == 'Disabled':
        return Mojo_In
    KeepMojo = []
    MergeMojo = []
    MergeSet = {}

    MergerContainer = []
    Sig=[]
    MergeTok = [[],{},{}]

    for Tok in Mojo_In:
        
        if Squeeze_Mojo == 'Keep Masks':
            if "mask" in Tok[1]:
                KeepMojo.append(Tok)
                continue
        lenTok = len(Tok[0])
        for i in range(lenTok):
            for key in Tok[0][i]:
                if len(MergerContainer) < (i+1):
                    MergerContainer.append({})
                    Sig.append({})
                if not key in MergerContainer[i]:
                    MergerContainer[i][key]=[]
                    sig=[]
                    tmpL = len(Tok[0][i][key][0][0])
                    for r in range(tmpL):
                        if type(Tok[0][i][key][0][0][r]) is int:
                            sig.append("int")
                        elif type(Tok[0][i][key][0][0][r]) is float:
                            sig.append("float")
                        elif type(Tok[0][i][key][0][0][r]) is complex:
                            sig.append("complex")
                    Sig[i][key] = sig

                MergerContainer[i][key].append(Tok[0][i][key])

    MClen = len(MergerContainer)
    if MClen > 0:
        for m in range(MClen):
            for key in MergerContainer[m]:
            
                cont = None
                for g in range(len(MergerContainer[m][key])):
                    npArr = np.array(MergerContainer[m][key][g], np.float32)
                    if cont is None:
                        cont = npArr
                    else:
                        cont = np.maximum(cont, npArr)
            
                if Merge_Mojo is not None:
                    lastInd = len(Merge_Mojo)-1
                    if m < len(Merge_Mojo[lastInd][0][m]):
                        if key in Merge_Mojo[lastInd][0][m]:
                            Merge_Mojo_npArr = np.array(Merge_Mojo[lastInd][0][m][key], np.float32)
                            tch = torch.from_numpy(cont)
                            MMtch = torch.from_numpy(Merge_Mojo_npArr)
                            cont = torch.lerp(MMtch,tch,Strength).numpy()

                if len(MergeTok[0]) < (m+1):
                    MergeTok[0].append({})

                lister=[]
                for element in cont[0]:
                    L=[]
                    for T in range(len(element)):
                        match Sig[i][key][T]:
                            case "int":
                                L.append(int(round(element[T],6)))
                            case "float":
                                L.append(float(element[T]))
                            case "complex":
                                L.append(complex(element[T]))
                    
                    lister.append(L)

                MergeTok[0][m][key] = [lister]

        MergeTok[2] = Mojo_In[len(Mojo_In)-1][2]
        KeepMojo.append(MergeTok)

    return KeepMojo

def TakeIn(OutM, step_start, step_end, Mojo_Intake, Squeeze_Mojo_Intake, Intake_Strength, setInfo):
    outMojo = []
    outMojo = OutM

    if Mojo_Intake is not None and Intake_Strength>0.0:
        if Intake_Strength==1.0:
            InMojo_Intake = SqueezeMojo(Mojo_Intake, Squeeze_Mojo_Intake)
            SetMojo(InMojo_Intake, setInfo["guidance"], setInfo["width"], setInfo["height"], setInfo["crop_w"], setInfo["crop_h"], setInfo["target_width"], setInfo["target_height"])

            for a in InMojo_Intake:
                outMojo.append(a)
        else:
            InMojo_Intake = SqueezeMojo(Mojo_Intake, Squeeze_Mojo_Intake, outMojo, Intake_Strength)
            SetMojo(InMojo_Intake, setInfo["guidance"], setInfo["width"], setInfo["height"], setInfo["crop_w"], setInfo["crop_h"], setInfo["target_width"], setInfo["target_height"])

            inStart = InMojo_Intake[len(InMojo_Intake)-1][2]["start_percent"]
            inEnd = InMojo_Intake[len(InMojo_Intake)-1][2]["end_percent"]

            startA = min(inStart, step_start)
            endA = max(inEnd,step_end)

            startOv = max(inStart, step_start)
            endOv = min(inEnd,step_end)

            OM = []

            if startOv-endOv<0:
                for a in InMojo_Intake:
                    OM.append(a)
                    OM[len(OM)-1][2]["start_percent"] = startOv
                    OM[len(OM)-1][2]["end_percent"] = endOv
                    
                if step_start<startOv:
                    for a in outMojo:
                        OM.append(outMojo[a])
                        OM[len(OM)-1][2]["start_percent"] = step_start
                        OM[len(OM)-1][2]["end_percent"] = startOv
                elif inStart<startOv:
                    MI = SqueezeMojo(Mojo_Intake, Squeeze_Mojo_Intake)
                    SetMojo(MI, setInfo["guidance"], setInfo["width"], setInfo["height"], setInfo["crop_w"], setInfo["crop_h"], setInfo["target_width"], setInfo["target_height"])
                    for a in MI:
                        OM.append(a)
                        OM[len(OM)-1][2]["start_percent"] = inStart
                        OM[len(OM)-1][2]["end_percent"] = startOv

                if step_end>endOv:
                    for a in outMojo:
                        OM.append(outMojo[a])
                        OM[len(OM)-1][2]["start_percent"] = endOv
                        OM[len(OM)-1][2]["end_percent"] = endA
                elif inEnd>endOv:
                    MI = SqueezeMojo(Mojo_Intake, Squeeze_Mojo_Intake)
                    SetMojo(MI, setInfo["guidance"], setInfo["width"], setInfo["height"], setInfo["crop_w"], setInfo["crop_h"], setInfo["target_width"], setInfo["target_height"])
                    for a in MI:
                        OM.append(a)
                        OM[len(OM)-1][2]["start_percent"] = endOv
                        OM[len(OM)-1][2]["end_percent"] = inEnd

                outMojo=OM

            else:
                for a in InMojo_Intake:
                    outMojo.append(a)
    return outMojo

def makeCond(clip, ClipT, Buffer, choice):

    mask = None
    srt = 1.0
    setarea = False
    compatibe = True

    if "mask" in Buffer[1]:
        mask = Buffer[1]["mask"]
        srt = Buffer[1]["mask_strength"]
        setarea = Buffer[1]["set_area_to_bounds"]

    Tok = {}

    def clipDetect(Clip_T, Buffer, indPref, AllArr):
        Tok = {}
        ind = 0
        SatArr = Clip_T.copy()
        Msg = {}

        for c in Buffer:
            for ck in c.keys():
                if not ck in AllArr:
                    AllArr.append(ck)

            for key in Clip_T:
                if key in c.keys():
                    if key in Tok.keys():
                        if ind == indPref:
                            Tok[key] = c[key]
                            Msg[key] = f', {key} from {NrToCh(ind)}'
                    else:
                        Tok[key] = c[key]
                        Msg[key] = f', {key} from {NrToCh(ind)}'
                        SatArr.remove(key)
            ind+=1

        Biff = ""
        if len(SatArr) > 0:
            Biff += " has no matching encoder signature and has been excluded. Main_clip invalid."
            compatibe = False

        for c in Msg:
            Biff += Msg[c] 
        return Tok, Biff, AllArr
            
    AllArr = []

    Tok, Buff, AllArr = clipDetect(ClipT, Buffer[0], ChToNr(choice), AllArr)          
            
    C = []
    M = {}
    if compatibe:
        C, M = condition(clip, Tok, 
                                Buffer[2]["guidance"], 
                                Buffer[2]["width"], 
                                Buffer[2]["height"], 
                                Buffer[2]["crop_w"], 
                                Buffer[2]["crop_h"], 
                                Buffer[2]["target_width"],
                                Buffer[2]["target_height"],
                                Buffer[2]["start_percent"],
                                Buffer[2]["end_percent"],
                                mask, 
                                srt, 
                                "mask bounds" if setarea else "default")
    return C, M, Buff, AllArr

def CondMojo(Main_clip, Buffer, ClipT, Clip_Choice, Info, Name="Mojo"):
    
    inx = 0
    C = []
    Masks = []

    for borfer in Buffer:
        Info += f'\n\n{Name} {inx}'
        outC_Pos, outMask, outInfo, outTE = makeCond(Main_clip, ClipT, borfer, Clip_Choice)
        C += outC_Pos
        if "mask" in outMask:
            Masks.append(outMask["mask"])
        Info += outInfo
        Info += f' {outTE}'
        inx+=1

    return C, Masks, Info

def MaskMojo(Buffer):
    Masks = []
    for borfer in range(len(Buffer)):
        if "mask" in Buffer[borfer][1]:
            Masks.append(Buffer[borfer][1]["mask"])

    return Masks

class MojoLoader:
    full_output_folder = os.path.join(folder_paths.get_output_directory(), 'Hellkars/Mojos')
    flileList = []
    if os.path.exists(full_output_folder):
        flileList = os.listdir(full_output_folder)

    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                        "Main_clip": ("CLIP", {"tooltip":'Main Clip source'}),
                        "Output_Path": ("STRING", {"default": 'Hellkars/Mojos', "multiline": False, "tooltip":'Subfolder Path into "output"'}),
                        "Name": ("STRING", {"default": "ShareMy", "tooltip":'Mojo File Name'}),
                        "Clip_Choice": (['Main', 'Additional', 'Special'],{"default":'Main', "tooltip":'Biases the assembly towards a clip source. IF such source is provided. The only mandatory source is Main. Usage is heavily dependent on how the .mojo file was made.'}), 
                        "Conditioning_Output": (['Disabled','Mojo (Combine)'],{"default":'Mojo (Combine)', "tooltip":'How to construct the Conditioning Output'}),}}

    RETURN_TYPES = ('LATENT','CONDITIONING','CONDITIONING','MOJO','MOJO', 'STRING', 'MASK', 'MASK')
    RETURN_NAMES = ('Latents','Positive', 'Negative', 'Positive_Mojo', 'Negative_Mojo', 'Info', 'Positive_Masks', 'Negative_Masks')
    OUTPUT_IS_LIST = (False, False, False, False, False, False, True, True)
    OUTPUT_NODE = False

    OUTPUT_TOOLTIPS = ('Latents',
                      'Positive Conditionings, if not "Disabled"',
                      'Negative Conditionings, if not "Disabled"',
                      'Positive Mojo Flow',
                      'Negative Mojo Flow',
                      "Info String with useful information about the Mojo and it's assembly.",
                      'Positive Masks, if provided',
                      'Negative Masks, if provided')

    FUNCTION = 'energize'
    CATEGORY = "Hellrunner's/Mojo"
    DESCRIPTION = "Loads a .mojo file for further refinement or recreation of an image. The usage of the content may vary dependent on it's construction."

    @classmethod
    def IS_CHANGED(self, Main_clip, Output_Path, Name, Clip_Choice, Conditioning_Output):
        return getVer(os.path.join(self.full_output_folder, f'{Name}.mojo'))  

    @classmethod
    def VALIDATE_INPUTS(self, Main_clip, Output_Path, Name, Clip_Choice, Conditioning_Output):
        if not os.path.exists(os.path.join(self.full_output_folder, f'{Name}.mojo')):
            return "Invalid Mojo File: {}".format(Name)

        return True

    def energize(self, Main_clip, Output_Path, Name, Clip_Choice, Conditioning_Output):

        self.full_output_folder = os.path.join(folder_paths.get_output_directory(), Output_Path)
        self.flileList = os.listdir(self.full_output_folder)
        compatibe = True

        Buffer = {}

        file = open(os.path.join(self.full_output_folder, f'{Name}.mojo'), "rb")
        data = file.read()
        loaded = load(data)
        file.close()

        for Key in loaded.keys():

            TenK = loaded[Key]
            if '.' in Key:
                SplitArray = Key.split('.')

                if not SplitArray[0] in Buffer:
                    Buffer[SplitArray[0]] = []

                index = int(SplitArray[1])

                while len(Buffer[SplitArray[0]]) < (index+1):
                    Buffer[SplitArray[0]].append([[],{},{}])

                if SplitArray[2] == "Tik":
                    Tik_index = int(SplitArray[3])
                    while len(Buffer[SplitArray[0]][index][0]) < (Tik_index+1):
                        Buffer[SplitArray[0]][index][0].append({})
                    Buffer[SplitArray[0]][index][0][Tik_index][SplitArray[4]] = json.loads("".join(map(chr, TenK)))

                if SplitArray[2] == "Mask":
                    if SplitArray[3] == "mask":
                        Buffer[SplitArray[0]][index][1][SplitArray[3]] = TenK
                    else:
                        Buffer[SplitArray[0]][index][1][SplitArray[3]] = json.loads("".join(map(chr, TenK)))

                if SplitArray[2] == "Extra":
                    Buffer[SplitArray[0]][index][2][SplitArray[3]] = json.loads("".join(map(chr, TenK)))  
                    
            else:
                if Key == "Info":
                    Buffer[Key] = json.loads("".join(map(chr, TenK)))
                else:
                    Buffer[Key] = TenK

        C_Pos=[]
        C_Neg=[]
        Buffer["PosMasks"]=[]
        Buffer["NegMasks"]=[]
        if Conditioning_Output != "Disabled":
            ClipT = ClipTest(Main_clip)
            C_Pos, Buffer["PosMasks"], Buffer["Info"] = CondMojo(Main_clip, Buffer["Pos"], ClipT, Clip_Choice, Buffer["Info"],"Positive")
            C_Neg, Buffer["NegMasks"], Buffer["Info"] = CondMojo(Main_clip, Buffer["Neg"], ClipT, Clip_Choice, Buffer["Info"],"Negative")
        else:
            Buffer["PosMasks"] = MaskMojo(Buffer["Pos"])
            Buffer["NegMasks"] = MaskMojo(Buffer["Neg"])


        Buffer['Info'] += f'\n\nMojo File iteration {Buffer["Ver"][0][0]}\n\n'

        return ({"samples": Buffer['Lnt']}, C_Pos, C_Neg, Buffer['Pos'], Buffer['Neg'], Buffer['Info'], Buffer["PosMasks"], Buffer["NegMasks"])

class MojoMaker:
    @classmethod
    def INPUT_TYPES(s):
        return{
            "required":{
                "Main_clip": ("CLIP", {"tooltip":'Main Clip source'}),},
            "optional":{
                "Alternative_clip": ("CLIP", {"tooltip":'Alternative Clip source. Enrich the possibilities of your Mojo'}),
                "Special_clip": ("CLIP", {"tooltip":'YES... MORE.. more clip... or t5, who knows?!'}),
                "Mojo_Intake": ("MOJO", {"default": None, "tooltip":'Join the incoming Mojo Flow into the output'}),
                "Phrases": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True, "tooltip":'l,hydit_clip,h - Normal Prompt. Comma separated, short phrases. Object and detail focused. (Local, Character)'}),
                "Tags": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True, "tooltip":'g - Simplified, tag-based prompting. Strong conceptional influence. (General, Action)'}),
                "Sentences": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True, "tooltip":'t5xxl,pile_t5xl,mt5xl - Go nuts. Write an epic novel. But set the mood... the t5s want dinner first. ^^ Might be generalized and reliant on lower clip concepts.'}), 
                "Mask_set_cond_area": (["default", "mask bounds"], {"default":"default", "tooltip":'Mask area behavior. "default": black pixel = (tag:0) - merges well with other areas  "mask bounds": black pixel = tag removed... off  gone from vocabulary - hard separation'}),
                "Mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip":'Mask strength. leave at 1.0 if you are using Mask-Maps'}),
                "guidance": ("FLOAT", {"default": 3.5, "min": -100.0, "max": 100.0, "step": 0.1, "tooltip":'Flux specific. Parameter for simulated guidance based on conditioning'}),
                "width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Set the current image resolution to match the Mojo perfectly.'}),
                "height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Set the current image resolution to match the Mojo perfectly.'}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Crop Factor Width'}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Crop Factor Height'}),
                "target_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Set the target image resolution after upscale.'}),
                "target_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Set the target image resolution after upscale.'}),
                "step_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip":'Step influence clamp start. Starts at 0-1. <0 Disabled'}),
                "step_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip":'Step influence clamp end. Ends at 0-1. <0 Disabled'}),
                "Squeeze_Mojo_Intake": (['Disabled', 'Squeeze + Toss', 'Keep Masks'],{"default":'Disabled', "tooltip":'Merge all Intake Mojo to the smallest possible count. Only maskless Mojo can be squeezed.'}),
                "Intake_Strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":'Mixing strength of the squeezed Intake.'}),
                "Conditioning_Output": (['Disabled','Mojo', 'Mojo + Intake (Combine)'],{"default":'Mojo', "tooltip":'How to construct the Conditioning Output. "Mojo + Intake (Combine)" is equivalent to the "Mojo (Combine)" Loader method'}),
                "mask": ("MASK", {"tooltip":'Weave a Mask into the Mojo Flow'})}}

    RETURN_TYPES = ('MOJO','CONDITIONING')
    RETURN_NAMES = ("Mojo", "Conditioning")
    FUNCTION = "MojoMe"

    OUTPUT_TOOLTIPS = ('Mojo Flow',
                      'Main_clip Conditioning, if not "Disabled"')

    CATEGORY = "Hellrunner's/Mojo"
    DESCRIPTION = "Weaves the clips and prompts into a unified Mojo Flow and outputs the Main_clip conditioning. Ensures feature complete usage of common models."

    @classmethod
    def VALIDATE_INPUTS(s, Main_clip=None, Alternative_clip=None, Special_clip=None, Mojo_Intake=None, Phrases="", Tags="", Sentences="", guidance=3.5, width=1024.0, height=1024.0, crop_w=0, crop_h=0, target_width=1024.0, target_height=1024.0, step_start=-1.0, step_end=-1.0, mask=None, Mask_strength=1.0, Mask_set_cond_area = "default", Squeeze_Mojo_Intake='Disabled', Intake_Strength=1.0, Conditioning_Output='Mojo'):
        if Phrases == "" and Tags == "" and Sentences == "":
            return "Nothing to convert"
        return True

    def MojoMe(self, Main_clip=None, Alternative_clip=None, Special_clip=None, Mojo_Intake=None, Phrases="", Tags="", Sentences="", guidance=3.5, width=1024.0, height=1024.0, crop_w=0, crop_h=0, target_width=1024.0, target_height=1024.0, step_start=-1.0, step_end=-1.0, mask=None, Mask_strength=1.0, Mask_set_cond_area = "default", Squeeze_Mojo_Intake='Disabled', Intake_Strength=1.0, Conditioning_Output='Mojo'):

        if Phrases == "":
            if Sentences == "":
                Phrases = Tags
            else:
                Phrases = Sentences
        if Sentences == "":
            Sentences = Phrases
        if Tags == "":
            Tags = Phrases

        def makeTik(clip, p,t,s):
            Tik = {}
            valid_l = ['l','hydit_clip', 'h']
            valid_g = ['g']
            valid_t5 = ['t5xxl','pile_t5xl','mt5xl']

            toDo = []

            tokens=clip.tokenize(p)
            for key in tokens.keys():
                if key in valid_l:
                    Tik[key] = tokens[key]
                    continue
                toDo.append(key)

            for typ in toDo:
                if typ in valid_g:
                    Tik[typ]=clip.tokenize(t)[typ]

                elif typ in valid_t5:
                    Tik[typ]=clip.tokenize(s)[typ]
            return Tik
        
        Tok = [[],{},{}]
        Tok[0].append(makeTik(Main_clip, Phrases, Tags, Sentences))

        if Alternative_clip is not None:
            Tok[0].append(makeTik(Alternative_clip, Phrases, Tags, Sentences))
        if Special_clip is not None:
            Tok[0].append(makeTik(Special_clip, Phrases, Tags, Sentences))

        C = []
        if Conditioning_Output == 'Mojo':
            start=0.0
            end=1.0
            if step_start >= 0.0:
                start = step_start
            if step_end >= 0.0:
                end = step_end
            C, Tok[1] = condition(Main_clip, Tok[0][0], guidance, width, height, crop_w, crop_h, target_width, target_height, start, end, mask, Mask_strength, Mask_set_cond_area)
        else:
            C, Tok[1] = makeMaskSlot(mask, Mask_strength, Mask_set_cond_area)

        Tok[2]=makeSettings({}, guidance, width, height, crop_w, crop_h, target_width, target_height, step_start, step_end)

        outMojo = [Tok]

        setInfo={}
        setInfo["guidance"] = guidance
        setInfo["width"] = width
        setInfo["height"] = height
        setInfo["crop_w"] = crop_w
        setInfo["crop_h"] = crop_h
        setInfo["target_width"] = target_width
        setInfo["target_height"] = target_height
        
        outMojo = TakeIn(outMojo, step_start, step_end, Mojo_Intake, Squeeze_Mojo_Intake, Intake_Strength, setInfo)

        if Conditioning_Output == 'Mojo + Intake (Combine)':
            ClipT = ClipTest(Main_clip)
            C, Masks, Info = CondMojo(Main_clip, outMojo, ClipT, "Main", "")
        
        return (outMojo, C)

class AdjustMojo:
    @classmethod
    def INPUT_TYPES(s):
        return{
            "required":{
                "Mojo": ("MOJO", {"default": None, "tooltip":'Mojo Flow'}),},
            "optional":{
                "Main_clip": ("CLIP", {"tooltip":'Main Clip source. Needed if Conditioning is enabled'}),
                "Mojo_Intake": ("MOJO", {"default": None, "tooltip":'Join the incoming Mojo Flow into the output'}),
                "guidance": ("FLOAT", {"default": 3.5, "min": -100.0, "max": 100.0, "step": 0.1, "tooltip":'Flux specific. Parameter for simulated guidance based on conditioning'}),
                "width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Set the current image resolution to match the Mojo perfectly.'}),
                "height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Set the current image resolution to match the Mojo perfectly.'}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Crop Factor Width'}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Crop Factor Height'}),
                "target_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Set the target image resolution after upscale.'}),
                "target_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "tooltip":'SDXL and Pony specific. Set the target image resolution after upscale.'}),
                "step_start": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip":'Step influence clamp start. Starts at 0-1. <0 Disabled'}),
                "step_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip":'Step influence clamp end. Ends at 0-1. <0 Disabled'}),
                "Squeeze_Mojo": (['Disabled', 'Squeeze + Toss', 'Keep Masks'],{"default":'Disabled', "tooltip":'Merge all Mojo to the smallest possible count. Only maskless Mojo can be squeezed.'}),
                "Squeeze_Mojo_Intake": (['Disabled', 'Squeeze + Toss', 'Keep Masks'],{"default":'Disabled', "tooltip":'Merge all Intake Mojo to the smallest possible count. Only maskless Mojo can be squeezed.'}),
                "Intake_Strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":'Mixing strength of the squeezed Intake.'}),
                "Clip_Choice": (['Main', 'Additional', 'Special'],{"default":'Main', "tooltip":'Biases the assembly towards a clip source. IF such source is provided. The only mandatory source is Main. Usage is heavily dependent on how the Mojo was made.'}),
                "Conditioning_Output": (['Disabled','Mojo (Combine)', 'Intake (Combine)', 'Mojo + Intake (Combine)'],{"default":'Mojo (Combine)', "tooltip":'How to construct the Conditioning Output.'})}}

    RETURN_TYPES = ('MOJO','CONDITIONING','STRING')
    RETURN_NAMES = ("Mojo", "Conditioning", "Info")
    FUNCTION = "AdjustMe"

    OUTPUT_TOOLTIPS = ('Mojo Flow',
                      'Main_clip Conditioning, if not "Disabled"',
                      'Info String with useful information about the Mojo and the assembly of it.')

    CATEGORY = "Hellrunner's/Mojo"
    DESCRIPTION = "Inject new values into a Mojo flow and/or weave in another"

    def AdjustMe(self, Mojo, Main_clip=None, Mojo_Intake=None, guidance=3.5, width=1024.0, height=1024.0, crop_w=0, crop_h=0, target_width=1024.0, target_height=1024.0, step_start=-1.0, step_end=-1.0, Squeeze_Mojo='Disabled', Squeeze_Mojo_Intake='Disabled', Intake_Strength=1.0, Clip_Choice='Main', Conditioning_Output='Mojo (Combine)'):
        
        InMojo = SqueezeMojo(Mojo, Squeeze_Mojo)
        SetMojo(InMojo, guidance, width, height, crop_w, crop_h, target_width, target_height, step_start, step_end)

        outMojo = InMojo

        setInfo={}
        setInfo["guidance"] = guidance
        setInfo["width"] = width
        setInfo["height"] = height
        setInfo["crop_w"] = crop_w
        setInfo["crop_h"] = crop_h
        setInfo["target_width"] = target_width
        setInfo["target_height"] = target_height
        
        outMojo = TakeIn(outMojo, step_start, step_end, Mojo_Intake, Squeeze_Mojo_Intake, Intake_Strength, setInfo)

        C=[]
        Info=""
        if Conditioning_Output != "Disabled":
            if Main_clip is None:
                print('Adjust Mojo - Main_clip required')
            else:
                ClipT = ClipTest(Main_clip)
                match Conditioning_Output:
                    case 'Mojo (Combine)':
                        C, Masks, Info = CondMojo(Main_clip, InMojo, ClipT, Clip_Choice, Info)
                    case 'Intake (Combine)':
                        if InMojo_Intake is not None:
                            C, Masks, Info = CondMojo(Main_clip, InMojo_Intake, ClipT, Clip_Choice, Info)
                        else:
                            C, Masks, Info = CondMojo(Main_clip, InMojo, ClipT, Clip_Choice, Info)
                    case 'Mojo + Intake (Combine)':
                        C, Masks, Info = CondMojo(Main_clip, outMojo, ClipT, Clip_Choice, Info)
        return (outMojo, C, Info)

NODE_CLASS_MAPPINGS = {
    "MojoMaker": MojoMaker,
    "SaveMojo": SaveMojo,
    "MojoLoader": MojoLoader,
    "AdjustMojo": AdjustMojo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MojoMaker": "Mojo Maker",
    "SaveMojo": "Save Mojo",
    "MojoLoader": "Mojo Loader",
    "AdjustMojo": "Adjust Mojo",
}