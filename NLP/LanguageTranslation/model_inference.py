from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch 

class MachineTranslation():
    
    def __init__(self, use_cuda=True):

        self.supported_languages = ["english", "russian", "japanese", "portuguese", "arabic", "german", "french", "spanish", "chinese"]
        self.supported_algorithms = ["nllb", "mbart", "m2m"]

        
        if torch.cuda.is_available() and use_cuda:
            self.use_cuda = True
        else:
            self.use_cuda = False
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
            

        
        #NLLB params
        self.model_nllb = None #AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")    
        self.nllb_tokenizer_english = None
        self.nllb_tokenizer_russian = None
        self.nllb_tokenizer_japanese = None
        self.nllb_tokenizer_portuguese = None
        self.nllb_tokenizer_arabic = None
        self.nllb_tokenizer_german = None
        self.nllb_tokenizer_french = None
        self.nllb_tokenizer_spanish = None
        self.nllb_tokenizer_chinese = None
        
        #MBART params
        self.model_mbart = None #MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer_mbart = None #MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

        #M2M params
        self.model_m2m = None #M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.tokenizer_m2m = None #M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

        
        
    def translate(self, input_text, src_language="english", target_language="russian", algorithm="nllb"):
        # input_text: str # text of any length.
        # src_language: str # Source language code. ,
        # target_language: str # Target language code
        # algorithm: str # algorithms, possibilites: nllb, mbart, m2m
        
        
        
        
        if src_language not in self.supported_languages:
            print("src_language should be one of: ", self.supported_languages)
            return None
        
        if target_language not in self.supported_languages:
            print("target_language should be one of: ", self.supported_languages)
            return None
        
        if algorithm not in self.supported_algorithms:
            print("algorithm should be one of: ", self.supported_algorithms)
            return None
        
        if len( input_text.strip() ) == 0: #Empty input, nothing to do
            return ""
        
        self.input_text = input_text
        self.src_language = src_language
        self.target_language = target_language
        self.algorithm = algorithm
        
        
        if algorithm == "nllb":
            return self.run_nllb()
        elif algorithm == "mbart":
            return self.run_mbart()
        else:
            return self.run_m2m()
        
    def run_mbart(self):
        
        def get_mbart_code(language):
            target_code = "en_XX"
            if language == "english":
                target_code = "en_XX"
            elif language == "russian":
                target_code = "ru_RU"
            elif language == "japanese":
                target_code = "ja_XX"
            elif language == "portuguese":
                target_code = "pt_XX"
            elif language == "arabic":
                target_code = "ar_AR"
            elif language == "german":
                target_code = "de_DE"
            elif language == "french":
                target_code = "fr_XX"
            elif language == "spanish":
                target_code = "es_XX"
            elif language == "chinese":
                target_code = "zh_CN"
            return target_code
            
            
        
        if self.model_mbart == None:
            self.model_mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            if self.use_cuda:
                self.model_mbart = self.model_mbart.cuda()
            
        if self.tokenizer_mbart == None:
            self.tokenizer_mbart = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        
        output_text = ""  
        self.tokenizer_mbart.src_lang = get_mbart_code(self.src_language)
        target_language_code = get_mbart_code(self.target_language)
        try:
            encoded = self.tokenizer_mbart(self.input_text, return_tensors="pt").to(self.device)
            max_new_tokens = int( 1.5*len(encoded["input_ids"][0]) )
            generated_tokens = self.model_mbart.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                forced_bos_token_id=self.tokenizer_mbart.lang_code_to_id[target_language_code]
            )
            output_text = self.tokenizer_mbart.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        except:
            pass
        
        return output_text   
    
    def run_m2m(self):
        
        def get_m2m_code(language):
            target_code = "en"
            if language == "english":
                target_code = "en"
            elif language == "russian":
                target_code = "ru"
            elif language == "japanese":
                target_code = "ja"
            elif language == "portuguese":
                target_code = "pt"
            elif language == "arabic":
                target_code = "ar"
            elif language == "german":
                target_code = "de"
            elif language == "french":
                target_code = "fr"
            elif language == "spanish":
                target_code = "es"
            elif language == "chinese":
                target_code = "zh"
            return target_code
        

        if self.model_m2m == None:
            self.model_m2m = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
            if self.use_cuda:
                self.model_m2m = self.model_m2m.cuda()

            
        if self.tokenizer_m2m == None:
            self.tokenizer_m2m = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")    
        
        output_text = ""
        self.tokenizer_m2m.src_lang = get_m2m_code(self.src_language)
        target_language_code = get_m2m_code(self.target_language)
        try:
            encoded = self.tokenizer_m2m(self.input_text, return_tensors="pt").to(self.device)
            max_new_tokens = int( 1.5*len(encoded["input_ids"][0]) )
            generated_tokens = self.model_m2m.generate(**encoded, max_new_tokens=max_new_tokens, forced_bos_token_id=self.tokenizer_m2m.get_lang_id(target_language_code))
            output_text = self.tokenizer_m2m.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        except:
            pass
        
        return output_text
        
        
    def run_nllb(self):
        
        if self.model_nllb is None:
            self.model_nllb = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            if self.use_cuda:
                self.model_nllb = self.model_nllb.cuda()
        
        target_code = "eng_Latn"
        if self.target_language == "english":
            target_code = "eng_Latn"
        elif self.target_language == "russian":
            target_code = "rus_Cyrl"
        elif self.target_language == "japanese":
            target_code = "jpn_Jpan"
        elif self.target_language == "portuguese":
            target_code = "por_Latn"
        elif self.target_language == "arabic":
            target_code = "arb_Arab"
        elif self.target_language == "german":
            target_code = "deu_Latn"
        elif self.target_language == "french":
            target_code = "fra_Latn"
        elif self.target_language == "spanish":
            target_code = "spa_Latn"
        elif self.target_language == "chinese":
            target_code = "zho_Hant"

        output_text = ""
            
        if self.src_language == "english":
            if self.nllb_tokenizer_english == None:
                self.nllb_tokenizer_english = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn")

            try:
                inputs = self.nllb_tokenizer_english(self.input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                max_new_tokens = int( 1.5*len(inputs["input_ids"][0]) )
                translated_tokens = self.model_nllb.generate(
                    **inputs, forced_bos_token_id=self.nllb_tokenizer_english.lang_code_to_id[target_code], max_length=max_new_tokens
                )
                output_text = self.nllb_tokenizer_english.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            except:
                pass
            return output_text
        
        if self.src_language == "russian":
            if self.nllb_tokenizer_russian == None:
                self.nllb_tokenizer_russian = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="rus_Cyrl")

            try:
                inputs = self.nllb_tokenizer_russian(self.input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                max_new_tokens = int( 1.5*len(inputs["input_ids"][0]) )
                translated_tokens = self.model_nllb.generate(
                    **inputs, forced_bos_token_id=self.nllb_tokenizer_russian.lang_code_to_id[target_code], max_length=max_new_tokens
                )
                output_text = self.nllb_tokenizer_russian.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            except:
                pass
            return output_text

        if self.src_language == "japanese":
            if self.nllb_tokenizer_japanese == None:
                self.nllb_tokenizer_japanese = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="jpn_Jpan")

            try:
                inputs = self.nllb_tokenizer_japanese(self.input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                max_new_tokens = int( 1.5*len(inputs["input_ids"][0]) )
                translated_tokens = self.model_nllb.generate(
                    **inputs, forced_bos_token_id=self.nllb_tokenizer_japanese.lang_code_to_id[target_code], max_length=max_new_tokens
                )
                output_text = self.nllb_tokenizer_japanese.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            except:
                pass
            return output_text
        
        if self.src_language == "portuguese":
            if self.nllb_tokenizer_portuguese == None:
                self.nllb_tokenizer_portuguese = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="por_Latn")

            try:
                inputs = self.nllb_tokenizer_portuguese(self.input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                max_new_tokens = int( 1.5*len(inputs["input_ids"][0]) )
                translated_tokens = self.model_nllb.generate(
                    **inputs, forced_bos_token_id=self.nllb_tokenizer_portuguese.lang_code_to_id[target_code], max_length=max_new_tokens
                )
                output_text = self.nllb_tokenizer_portuguese.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            except:
                pass
            return output_text

        if self.src_language == "arabic":
            if self.nllb_tokenizer_arabic == None:
                self.nllb_tokenizer_arabic = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="arb_Arab")

            try:
                inputs = self.nllb_tokenizer_arabic(self.input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                max_new_tokens = int( 1.5*len(inputs["input_ids"][0]) )
                translated_tokens = self.model_nllb.generate(
                    **inputs, forced_bos_token_id=self.nllb_tokenizer_arabic.lang_code_to_id[target_code], max_length=max_new_tokens
                )
                output_text = self.nllb_tokenizer_arabic.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            except:
                pass
            return output_text

        if self.src_language == "german":
            if self.nllb_tokenizer_german == None:
                self.nllb_tokenizer_german = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="deu_Latn")

            try:
                inputs = self.nllb_tokenizer_german(self.input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                max_new_tokens = int( 1.5*len(inputs["input_ids"][0]) )
                translated_tokens = self.model_nllb.generate(
                    **inputs, forced_bos_token_id=self.nllb_tokenizer_german.lang_code_to_id[target_code], max_length=max_new_tokens
                )
                output_text = self.nllb_tokenizer_german.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            except:
                pass
            return output_text

        if self.src_language == "french":
            if self.nllb_tokenizer_french == None:
                self.nllb_tokenizer_french = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="fra_Latn")

            try:
                inputs = self.nllb_tokenizer_french(self.input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                max_new_tokens = int( 1.5*len(inputs["input_ids"][0]) )
                translated_tokens = self.model_nllb.generate(
                    **inputs, forced_bos_token_id=self.nllb_tokenizer_french.lang_code_to_id[target_code], max_length=max_new_tokens
                )
                output_text = self.nllb_tokenizer_french.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            except:
                pass
            return output_text
        
        if self.src_language == "spanish":
            if self.nllb_tokenizer_spanish == None:
                self.nllb_tokenizer_spanish = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="spa_Latn")

            try:
                inputs = self.nllb_tokenizer_spanish(self.input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                max_new_tokens = int( 1.5*len(inputs["input_ids"][0]) )
                translated_tokens = self.model_nllb.generate(
                    **inputs, forced_bos_token_id=self.nllb_tokenizer_spanish.lang_code_to_id[target_code], max_length=max_new_tokens
                )
                output_text = self.nllb_tokenizer_spanish.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            except:
                pass
            return output_text

        
        if self.src_language == "chinese":
            if self.nllb_tokenizer_chinese == None:
                self.nllb_tokenizer_chinese = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="zho_Hant")


            try:
                inputs = self.nllb_tokenizer_chinese(self.input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                max_new_tokens = int( 1.5*len(inputs["input_ids"][0]) )
                translated_tokens = self.model_nllb.generate(
                    **inputs, forced_bos_token_id=self.nllb_tokenizer_chinese.lang_code_to_id[target_code], max_length=max_new_tokens
                )
                output_text = self.nllb_tokenizer_chinese.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            except:
                pass
            return output_text
        
        
        return output_text