import folder_paths

class LoRABox:
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                        "LoRAName": (folder_paths.get_filename_list("loras"), {"default": "", "tooltip":'Encoder Name to be added.'}),
                        "Model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.1, "tooltip":'Model weight'}),
                        "TE": ("FLOAT", {"default": 0.96, "min": -100.0, "max": 100.0, "step": 0.1, "tooltip":'Text Encoder weight'}),
                        "Active": ("BOOLEAN", {"default": True, "label_on":"On", "label_off":"Off", "tooltip":'LoRA status on/off'}),
                }
                ,"optional":{
                        "LoRABox": ("LORABOX", {"default": None, "tooltip":'ClipBox'}),
                }}

    RETURN_TYPES = ('LORABOX',)
    RETURN_NAMES = ('LoRABox',)
    OUTPUT_IS_LIST = (False,)
    OUTPUT_NODE = False

    OUTPUT_TOOLTIPS = ('LoRABox',)

    FUNCTION = 'energize'
    CATEGORY = "Hellrunner's/Snip/Boxes"
    DESCRIPTION = "Creates or extends a LoRABox"

    def energize(self, LoRAName,Model,TE,Active, LoRABox=None):

        lbox = []

        if not LoRABox == None:
            lbox = LoRABox

        element = {}
        element['Name'] = LoRAName
        element['Model'] = Model
        element['TE'] = TE
        element['Active'] = Active

        lbox.append(element)

        return (lbox,)


NODE_CLASS_MAPPINGS = {
    "LoRABox": LoRABox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRABox": "LoRABox (HMN)",
}