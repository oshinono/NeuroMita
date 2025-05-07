# NeuroMita v0.011  
A mod where you get to interact with Mitas controlled by neural networks. Built with Python and C# MelonMod.  

Mod Server: https://discord.gg/Tu5MPFxM4P (Get help here!)  

![logomod3](https://github.com/user-attachments/assets/aea3ec44-c203-4d4a-a405-a09191188464)  

# Installation Guide  

### 0) MelonLoader:  
A universal Unity modding tool. May conflict with BepInEx-based mods.  

- Install via: https://melonwiki.xyz/#/?id=requirements (Version 0.6.6)  
- Or directly from: https://github.com/LavaGang/MelonLoader  

If you have **MelonLoader.Installer.exe**, select **Miside** to patch it for Melon-based mods.  
Ensure all dependencies (e.g., .NET 6.0) are installed: https://melonwiki.xyz/#/?id=requirements  

### 1) The Mod  
The mod consists of:  
- **Python files** (place anywhere, but keep them together)  
- **C# files**: `MitaAI.dll` and `assetbundle.test` (place in the `Mods` folder created by MelonLoader).  

Final structure should look like:  

```
Miside  
- Other Miside folders  
- Mods (Create if needed)  
  - MitaAI.dll  
  - assetbundle.test  

Any Separate Folder  
- _internal  
- libs  
- Prompts  
- NeuroMita.exe  
(For local voice generation, also include:)  
- Models  
- features.env  
- include  
- init_triton.bat  
- init.py  
```  

Future versions may include a launcher.  

Download releases here: https://github.com/VinerX/NeuroMita/releases  
Actual release: https://github.com/VinerX/NeuroMita/releases/download/v0.011/NeuroMita.0.011.MitaWorld.7z

**In-game controls:**  
- Press **Tab** to start typing.  
- Press **Enter** to send.  

### 2) Text Generation  
The mod supports multiple text-generation methods (tested options listed below).  

#### Free API Options:  
- **g4f** (No API keys needed)  
- **OpenRouter** (Free keys: https://openrouter.ai/settings/keys – rate-limited)  
- **io.net** (Free keys: https://ai.io.net/ai/api-keys – 500k tokens/day per model)  

#### Paid API Options:  
- **OpenRouter** (Wide model selection, pay-per-use)  

#### Local Generation:  
- **LM Studio** (https://lmstudio.ai, requires strong hardware, for advanced users only)  

**Note:** Gemini models often handle emotions better, while GPT-4o is more precise but less expressive.  

### Models (as of 05/05/2025)  
*(Subject to rapid change—check Discord for updates!)*  

#### **G4F (No API Keys)**  
Good for testing, but weaker models. Enable the checkbox in the settings. If base model does not work, use button and reload.

![img_1.png](ReadmeFiles/img_1.png)

Supported models (selectable via version input + restart):  
- `gemini-1.5-flash` (Most stable in 0.4.7.7)  
- `gpt-4o-mini`  
- `gpt-4o`  
- `gemini-2.0-flash`  
- `deepseek-chat`  
Full list: https://github.com/xtekky/gpt4free/blob/main/docs/providers-and-models.md  

#### **OpenRouter (Free/Paid Keys)**  
Get keys here: https://openrouter.ai/settings/keys  

![img_2.png](ReadmeFiles/img_2.png)

Recommended free models:  
- `google/gemini-2.0-pro-exp-02-05:free`  
- `deepseek/deepseek-chat:free` (Best for "Kind Mita")  
- `deepseek/deepseek-chat-v3-0324:free` (Hardcore mode)  

Semi-paid (requires balance but no usage cost):  
- `google/gemini-2.5-pro-exp-03-25`  

Full list: https://openrouter.ai/models?max_price=0  

#### **Ai.iO (500k tokens/day per model)**  
API: https://api.intelligence.io.solutions/api/v1/  
- `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`  
Full list: https://docs.io.net/reference/get-started-with-io-intelligence-api  

## 3) Voice Generation  
Two options: **Telegram bots** or **local generation**.  

### **Telegram**  
Uses your Telegram account (preferably a secondary one) as a bot. Requires `api_id` and `api_hash` (guide: https://core.telegram.org/api/obtaining_api_id).  

After setup, restart and enter the confirmation code sent to your Telegram account. If you have 2FA, enter it (invisible input).  

Available bots:  
- **@CrazyMitaAIbot** (Free, unstable)  
- **@silero_voice_bot** (Paid, 600 test characters)  

**Note:** Manually message the bots first to ensure connectivity.  

### **Local Voice Generation**  
Requires the `Models` folder and `features.env` (included in releases).  
Download: https://github.com/VinerX/NeuroMita/releases/download/v0.011/Models.7z
1. Enable voice generation.  
2. Select **Local** (requires `features.env`).  
3. Choose and configure a model.  
*(Initial setup may take time due to downloads.)*  
![img_5.png](ReadmeFiles/img_5.png)
---

### **Credits**  
**Developers:**  
- **VinerX**  
- **vlad2830** (C# & Python)  
- **Nelxi (distrane25)** (Voice input integration)  

**Local Voice Generation (Massive Contribution):**  
- **_atm4x**  

**Character Prompts:**  
- **Feanor (feanorqq)** & **Tkost** (Kind Mita)  
- **Josefummi** (Short-Haired Mita)  
- **gad991** (Cap Mita)  
- **depikoov** (Sweet Mita)  

**Animations (WIP):**  
- **JPAV**  

**Pull Requests & CrazyMitaBot Contact:**  
- **スノー (v1nn1ty)**  

**Testers (Brave Bug Hunters):**  
- **GermanPlaygroud**  

**Special Thanks:**  
- **Sutherex** (Introduced OpenRouter, organizational help)  
- **Dr. Couch Science** (Early tester, admin support)  
- **Romancho** (Idea organization, community moderation)  
- **FlyOfFly** (Unity advice, early text input help)  
- **LoLY3_0** (The cat on a watermelon 🍉)  
- **Mr. Sub** (Likely how you found this mod!)  
- **All early testers** (Especially **smarkloker**)  
- **KASTA**  

**Support the author (VinerX):** https://boosty.to/vinerx  
