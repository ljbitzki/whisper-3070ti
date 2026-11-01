# Whisper Web Local — transcrição de áudio com CUDA no Ubuntu

Aplicação de **transcrição de voz para texto inteiramente local**, com suporte a arquivos de áudio e captura de microfone pelo navegador. O processamento é realizado pelo [whisper.cpp](https://github.com/ggml-org/whisper.cpp), acelerado por uma GPU NVIDIA, e a interface web é implementada com **FastAPI + HTML/CSS/JavaScript**, sem PyTorch ou dependências Python para inferência.

Este documento registra **todo o processo de instalação**, desde o preparo do Ubuntu e a compilação do `whisper.cpp` com CUDA até a execução da interface web. Foi elaborado para o cenário de referência **Ubuntu/Kubuntu 26.04, NVIDIA RTX 3070 Ti (8 GB)**, mas os comandos podem ser adaptados a outras distribuições e GPUs.

> **Escopo:** os passos abaixo documentam a instalação que utilizamos. O código da interface está neste repositório; o `whisper.cpp`, os modelos GGML, o CUDA Toolkit e o driver NVIDIA são instalados separadamente. Não é necessário fazer a etapa de integração de ditado com outros aplicativos do desktop.

## Funcionalidades

- Transcrição local de arquivos como WAV, MP3, M4A, OGG, Opus, FLAC, WebM e MP4, com conversão automática pelo FFmpeg.
- Captura de microfone diretamente no Chrome ou LibreWolf, com transcrição progressiva em blocos.
- Inferência acelerada por NVIDIA CUDA; o modelo fica carregado no processo `whisper-server`.
- Seleção de modelos GGML multilíngues já disponíveis em disco, incluindo `small` e `large-v3-turbo`.
- Idioma português como padrão (`pt`), com suporte a autodetecção e outros idiomas.
- Ajustes de temperatura, `beam_size`, `best_of`, limiar de ausência de fala, prompt de contexto e duração dos blocos do microfone.
- Visualização do texto, cópia e exportação para **TXT** ou **SRT**.
- Interface e API vinculadas a `127.0.0.1` por padrão, sem envio de áudio para serviços externos.

## Arquitetura

```text
Chrome / LibreWolf
  │
  ├── Seleção de arquivo ─────────────┐
  └── Microfone (Web Audio API) ──────┤
                                     ▼
                         FastAPI 127.0.0.1:7860
                           │         │
                           │         └── FFmpeg: WAV mono, 16 kHz, PCM 16 bits
                           ▼
                  whisper-server 127.0.0.1:18080
                     /inference   /load
                           │
                           ▼
                      whisper.cpp
                  modelo GGML multilíngue
                           │
                           ▼
                     NVIDIA CUDA
                      RTX 3070 Ti
                           │
                           ▼
                     Texto / TXT / SRT
```

A interface Python **não carrega modelos de IA**: ela controla o processo `whisper-server` do projeto C++ e envia o áudio por HTTP local. Ao selecionar outro modelo, utiliza `/load`; não cria um novo processo para cada trecho.

## 1. Pré-requisitos

Hardware e software utilizados:

| Componente | Ambiente de referência |
| --- | --- |
| Sistema | Ubuntu/Kubuntu 26.04 LTS, 64 bits |
| GPU | NVIDIA GeForce RTX 3070 Ti, 8 GB VRAM |
| Arquitetura CUDA da GPU | Compute capability **8.6** (`sm_86`) |
| Driver | Driver NVIDIA compatível com o CUDA Toolkit escolhido |
| Toolchain | CMake, compiladores C/C++, CUDA Toolkit (`nvcc`) |
| Áudio | FFmpeg e SDL2 (SDL2 é necessário para o `whisper-stream` de terminal) |
| Interface | Python 3 com `venv`, Chrome ou LibreWolf recentes |

Os exemplos adotam os diretórios:

```text
~/IA/
├── whisper.cpp/           # Código, modelos e binários nativos (projeto externo)
│   ├── models/
│   └── build-cuda/
└── whisper-web-local/     # Este repositório
    ├── app.py
    ├── requirements.txt
    ├── run.sh
    └── static/
```

Se preferir outro diretório para `whisper.cpp`, utilize a variável `WHISPER_DIR` ao executar a interface.

## 2. Instalação do driver NVIDIA

Primeiro, **verifique o driver existente**:

```bash
nvidia-smi
```

Se a RTX 3070 Ti for exibida normalmente, **não reinstale o driver**. Caso o comando falhe porque o driver ainda não está instalado, em uma instalação padrão do Ubuntu é possível utilizar:

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

Após reiniciar, execute novamente `nvidia-smi`.

> O campo `CUDA Version` de `nvidia-smi` informa a capacidade suportada pelo **driver**; a versão do **CUDA Toolkit** instalado deve ser verificada com `nvcc --version`.

## 3. Dependências do sistema e CUDA Toolkit

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  ffmpeg \
  curl \
  wget \
  libsdl2-dev \
  python3-venv
```

No Ubuntu 26.04, o CUDA Toolkit está disponível nos repositórios oficiais:

```bash
sudo apt install -y cuda-toolkit
```

Verifique:

```bash
nvidia-smi
nvcc --version
```

Se `nvcc` não estiver no `PATH`, confira primeiro onde foi instalado:

```bash
command -v nvcc || find /usr/local/cuda* /usr/bin -maxdepth 3 -name nvcc 2>/dev/null
```

Se estiver em `/usr/local/cuda/bin/nvcc`, ajuste o `PATH` para a sessão atual:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
```

Para tornar a configuração persistente no Bash, adicione a linha ao `~/.bashrc` (por exemplo, usando `vim ~/.bashrc`). A instalação pode disponibilizar o `nvcc` diretamente em `/usr/bin`; nesse caso, o ajuste não é necessário.

> Não misture indiscriminadamente pacotes de CUDA de múltiplos repositórios. Se a compilação indicar incompatibilidade com o driver, verifique as versões instaladas antes de atualizar qualquer componente.

## 4. Obter e compilar o whisper.cpp com CUDA

```bash
mkdir -p ~/IA
cd ~/IA

git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
```

Configure a compilação:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DWHISPER_SDL2=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

Compile:

```bash
cmake --build build-cuda --config Release -j 4
```

A arquitetura `86` corresponde à RTX 3070 Ti (Ampere). Para outra GPU, verifique o número adequado na documentação da NVIDIA.

Confira os executáveis:

```bash
ls -lh build-cuda/bin/whisper-cli \
       build-cuda/bin/whisper-stream \
       build-cuda/bin/whisper-server
```

Se o `whisper-server` não tiver sido gerado:

```bash
cmake --build build-cuda --target whisper-server -j 4
```

E confira as opções utilizadas:

```bash
grep -E 'GGML_CUDA:|WHISPER_SDL2:|CMAKE_CUDA_ARCHITECTURES:' \
  build-cuda/CMakeCache.txt
```

O esperado é encontrar `GGML_CUDA=ON`, `WHISPER_SDL2=ON` e arquitetura `86`.

**Importante:** os arquivos em `build-cuda/` são binários gerados no computador; não é necessário versioná-los neste repositório.

## 5. Baixar os modelos GGML multilíngues

Ainda em `~/IA/whisper.cpp`:

```bash
sh models/download-ggml-model.sh small
sh models/download-ggml-model.sh large-v3-turbo
```

Opcionalmente, para comparar qualidade de transcrição:

```bash
sh models/download-ggml-model.sh medium
sh models/download-ggml-model.sh large-v3
```

Confira:

```bash
ls -lh models/ggml-*.bin
```

| Modelo | Uso sugerido | Tamanho de download aproximado |
| --- | --- | ---: |
| `small` | Microfone, menor latência | 466 MB |
| `medium` | Testes de precisão | 1,5 GB |
| `large-v3-turbo` | Equilíbrio entre qualidade e velocidade | 1,6 GB |
| `large-v3` | Comparações com foco em qualidade | 3,1 GB |

O uso de VRAM é superior ao tamanho do arquivo do modelo e depende dos buffers utilizados. A interface procura arquivos `ggml-*.bin` no diretório `models/` e omite modelos com nome terminado em `.en.bin`.

> Para português brasileiro, utilize **modelos multilíngues** e `-l pt`. Modelos terminados em `.en` são voltados ao inglês.

## 6. Validar a transcrição de um arquivo no terminal

Primeiro, converta uma gravação para WAV mono em 16 kHz, PCM de 16 bits:

```bash
cd ~/IA/whisper.cpp

ffmpeg -i /caminho/para/reuniao.mp3 \
  -ar 16000 -ac 1 -c:a pcm_s16le reuniao.wav
```

Transcreva com o modelo `large-v3-turbo`:

```bash
mkdir -p ~/Transcricoes

./build-cuda/bin/whisper-cli \
  -m models/ggml-large-v3-turbo.bin \
  -f reuniao.wav \
  -l pt \
  -t 4 \
  -otxt -osrt \
  -of "$HOME/Transcricoes/reuniao"
```

Arquivos esperados:

```text
~/Transcricoes/reuniao.txt
~/Transcricoes/reuniao.srt
```

Em outro terminal, acompanhe a GPU durante a execução:

```bash
watch -n 0.5 nvidia-smi
```

A inicialização do backend deverá mencionar CUDA e a RTX 3070 Ti. Para um teste comparativo **com o mesmo áudio e o mesmo modelo**, desative o uso da GPU adicionando `-ng`:

```bash
./build-cuda/bin/whisper-cli \
  -m models/ggml-large-v3-turbo.bin \
  -f reuniao.wav \
  -l pt \
  -ng
```

Observe os tempos informados ao final de cada execução. A comparação de desempenho deve ser realizada com parâmetros equivalentes.

## 7. Validar a captura do microfone no terminal

O `whisper-stream` utiliza SDL2 e permite testar a captura independentemente da aplicação web.

**Modo contínuo**, com janelas de 5 segundos e atualização a cada segundo:

```bash
cd ~/IA/whisper.cpp

./build-cuda/bin/whisper-stream \
  -m models/ggml-small.bin \
  -l pt \
  -t 4 \
  --step 1000 \
  --length 5000
```

**Modo com VAD simples**, disparado por atividade de voz:

```bash
./build-cuda/bin/whisper-stream \
  -m models/ggml-small.bin \
  -l pt \
  -t 4 \
  --step 0 \
  --length 10000 \
  -vth 0.6
```

O `--step 0` habilita o modo de VAD básico do exemplo nativo, enquanto `-vth` define o limiar de detecção. Isso não deve ser confundido com um modelo VAD especializado.

Se houver mais de um microfone, consulte os dispositivos indicados pelo `whisper-stream` e selecione o índice adequado com `-c N` (por exemplo, `-c 1`).

> No modo contínuo do terminal, uma linha pode ser reescrita, e o mesmo trecho pode aparecer mais de uma vez. Esse comportamento não representa necessariamente uma duplicação do arquivo transcrito.

## 8. Instalar a interface web em um venv

O diretório `~/IA/whisper-web-local` deve conter os arquivos deste repositório:

```text
whisper-web-local/
├── README.md
├── app.py
├── requirements.txt
├── run.sh
└── static/
    ├── index.html
    ├── style.css
    ├── app.js
    └── mic-worklet.js
```

Se ainda não tiver compilado o servidor:

```bash
cd ~/IA/whisper.cpp
cmake --build build-cuda --target whisper-server -j 4
```

No diretório da interface:

```bash
cd ~/IA/whisper-web-local
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
chmod +x run.sh
```

**Inicie com:**

```bash
./run.sh
```

Acesse no mesmo computador:

**http://127.0.0.1:7860**

O `run.sh` inicia o Uvicorn/FastAPI, que por sua vez inicializa automaticamente o `whisper-server` na porta interna `18080` e carrega um modelo GGML. **Não inicie outro `whisper-server` manualmente na mesma porta.**

O modelo inicial é escolhido nesta ordem: `small`, `large-v3-turbo`, `medium`; se nenhum desses existir, é utilizado o primeiro modelo multilíngue encontrado.

### Configurações opcionais

Se o `whisper.cpp` estiver em outro diretório:

```bash
WHISPER_DIR=/outro/caminho/whisper.cpp ./run.sh
```

Se a porta interna `18080` estiver ocupada:

```bash
WHISPER_SERVER_PORT=18081 ./run.sh
```

A interface permanece em `127.0.0.1:7860` porque esse endereço está definido no `run.sh`; para alterá-lo, edite o script. **Não altere o host para `0.0.0.0` sem adicionar autenticação e controles de segurança.**

Para encerrar ambos os processos, utilize `Ctrl+C` no terminal que executa `./run.sh`.

## 9. Como utilizar a interface

### Arquivos gravados

1. Abra `http://127.0.0.1:7860`.
2. Escolha um modelo GGML (por exemplo, `large-v3-turbo`).
3. Mantenha **Português** como idioma, ou selecione outro idioma.
4. Na aba **Arquivo**, selecione a gravação e clique em **Transcrever arquivo**.
5. Revise o texto e escolha **Copiar**, **Salvar TXT** ou **Salvar SRT**.

O limite de upload implementado nesta versão é **300 MB** por arquivo. O FFmpeg extrai a primeira trilha de áudio (`0:a:0`) e converte a entrada para WAV mono 16 kHz antes de encaminhá-la ao motor nativo.

### Microfone

1. Abra a aba **Microfone** e selecione o modelo (comece com `small`).
2. Escolha uma janela de **5, 8, 12 ou 20 segundos** (padrão: 8 s).
3. Clique em **Iniciar gravação** e autorize o acesso ao microfone no navegador.
4. O texto será acrescentado progressivamente à tela após o processamento de cada bloco.
5. Clique em **Parar** para processar as amostras restantes e, se desejar, exportar o resultado.

A captura é feita com **Web Audio API/AudioWorklet**. A interface usa blocos não sobrepostos; portanto, o texto pode perder precisão em palavras atravessadas pelo limite entre blocos. **Não é streaming palavra por palavra** e **não há VAD na captura web desta versão**. O modo VAD do `whisper-stream` descrito na seção anterior é independente.

O navegador exige `localhost`/`127.0.0.1` (ou HTTPS em cenários remotos) para a API de microfone.

### Parâmetros disponíveis

| Opção | Finalidade |
| --- | --- |
| Modelo | Escolhe o modelo GGML no disco; pode provocar recarregamento na GPU |
| Idioma | Usa `pt` por padrão; `auto` tenta detectar |
| Temperatura | Ajusta a estratégia de decodificação; `0` é um bom ponto de partida |
| Beam size | Largura da busca, quando aplicável; `-1` utiliza o padrão do motor |
| Best of | Número de candidatos em estratégias de amostragem aplicáveis |
| Limiar sem fala | Critério do Whisper para classificar trechos como ausência de fala |
| Vocabulário de contexto | Prompt opcional com nomes próprios e termos técnicos |
| Janela do microfone | Duração de cada bloco enviado para inferência |

Um prompt de contexto pode conter termos como `Huawei`, `Kea DHCP`, `VLAN`, `NetBox` e `pfSense`. Ele **não treina** o modelo nem garante a grafia desses termos.

## 10. Logs, API e diagnóstico

Em outro terminal:

```bash
# Estado da interface e lista de modelos
curl -s http://127.0.0.1:7860/api/status | python3 -m json.tool

# Saúde do motor nativo
curl -s http://127.0.0.1:18080/health

# GPU e VRAM
watch -n 0.5 nvidia-smi

# Mensagens do whisper-server
tail -f ~/IA/whisper-web-local/whisper-server.log
```

Principais endpoints internos:

| Serviço | Endpoint | Uso |
| --- | --- | --- |
| FastAPI | `GET /` | Interface web |
| FastAPI | `GET /api/status` | Estado e modelos instalados |
| FastAPI | `POST /api/transcribe` | Recebe áudio e opções da interface |
| whisper-server | `GET /health` | Estado do motor |
| whisper-server | `POST /inference` | Inferência de áudio |
| whisper-server | `POST /load` | Troca do modelo GGML |

O aplicativo serializa as requisições de inferência e troca de modelo usando um bloqueio assíncrono, para evitar a mistura de modelos durante o processamento.

### Problemas frequentes

| Sintoma | O que verificar |
| --- | --- |
| `nvidia-smi` não funciona | Instalação/carregamento do driver NVIDIA |
| `nvcc: command not found` | CUDA Toolkit instalado e `PATH` do compilador |
| Erro do CMake relacionado a CUDA | Versões de driver/toolkit, compilador e `GGML_CUDA=ON` |
| Transcrição usa apenas CPU | Cache CMake e inicialização do backend; não utilizar `-ng` |
| `whisper-stream` não existe | `libsdl2-dev` e `WHISPER_SDL2=ON` na compilação |
| `whisper-server` não encontrado | Executar `cmake --build build-cuda --target whisper-server -j 4` |
| Interface não inicia | Verificar `.venv`, `requirements.txt`, `run.sh` e o log |
| Porta `18080` ocupada | Encerrar processo anterior ou usar `WHISPER_SERVER_PORT` |
| Erro 413 | Arquivo maior que o limite de 300 MB |
| Erro 415 | Extensão não aceita pelo backend |
| Microfone não funciona | Permissão do navegador, origem `localhost` e entrada selecionada no PipeWire |
| Texto do microfone com cortes | Aumentar a janela; a versão atual não usa sobreposição nem VAD web |
| Primeiro processamento demora | Carregamento do modelo; modelos maiores consomem mais VRAM |

Para inspecionar as portas locais:

```bash
ss -ltnp | grep -E ':7860|:18080'
```

## 11. Segurança e privacidade

- A aplicação executa em **loopback** (`127.0.0.1`) por padrão e não exige conexão à Internet durante o reconhecimento, após a instalação dos pacotes e o download dos modelos.
- Não inicie a interface nem o motor nativo como `root`.
- Os uploads são processados em diretórios temporários e removidos ao fim da requisição; o aplicativo não mantém uma biblioteca persistente das gravações ou transcrições.
- O texto é exibido no navegador e deve ser exportado explicitamente para persistir como arquivo de saída.
- A interface **não implementa autenticação**, gerenciamento multiusuário nem endurecimento para acesso público. Não exponha as portas `7860` e `18080` à rede sem uma revisão de segurança.
- FFmpeg, entradas fornecidas pelo usuário e o endpoint de troca de modelo exigem atenção especial caso a arquitetura seja adaptada para acesso por terceiros.

## 12. Atualização e versionamento

Os componentes possuem ciclos de atualização diferentes:

- **Este repositório:** `app.py`, `run.sh`, `requirements.txt`, arquivos `static/` e documentação.
- **Projeto externo:** `~/IA/whisper.cpp`, que deve ser atualizado e recompilado separadamente.
- **Sistema operacional:** driver NVIDIA, CUDA Toolkit, FFmpeg e SDL2.
- **Modelos:** arquivos binários GGML baixados por scripts oficiais.

Para identificar precisamente a revisão do motor usada nesta instalação:

```bash
git -C ~/IA/whisper.cpp rev-parse HEAD
```

Registre esse hash na documentação de implantação caso seja importante repetir exatamente a mesma versão. O `requirements.txt` utiliza intervalos de versões, não um arquivo de dependências totalmente congeladas; para reprodutibilidade estrita, mantenha também uma relação de versões efetivamente instaladas.

Um `.gitignore` recomendado para o repositório da interface:

```gitignore
.venv/
__pycache__/
*.py[cod]
*.log
.pytest_cache/
.env
.DS_Store
```

**Não versione** dados privados de áudio, transcrições, modelos GGML ou binários compilados, salvo se houver uma necessidade explícita e os direitos de distribuição estiverem claros.

## 13. Referências

- [whisper.cpp — projeto oficial](https://github.com/ggml-org/whisper.cpp)
- [whisper.cpp — captura contínua pelo microfone](https://github.com/ggml-org/whisper.cpp/tree/master/examples/stream)
- [whisper.cpp — servidor HTTP](https://github.com/ggml-org/whisper.cpp/tree/master/examples/server)
- [Ubuntu 26.04 — notas de versão e suporte ao CUDA Toolkit](https://documentation.ubuntu.com/release-notes/26.04/summary-for-lts-users/)
- [FastAPI — documentação oficial](https://fastapi.tiangolo.com/)
- [FFmpeg — documentação oficial](https://ffmpeg.org/documentation.html)

---

**Resultado esperado:** transcrição local em português, por arquivo ou microfone, usando um navegador e a RTX 3070 Ti com CUDA. A interface web reutiliza os modelos e o motor nativo já instalados, mantendo a camada Python restrita a um `venv` leve.
