import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
import random
import csv
from collections import deque
import pprint

# Define o dispositivo (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 1. PREPARAÇÃO DOS DADOS (com Concept Drift dinâmico Não-IID)
# -----------------------------
def create_dynamic_non_iid_data(dataset, num_clients=100, alpha_sequence=[0.3, 1.0, 0.1], drift_intervals=[50, 100, 200]):
    if not alpha_sequence:
        print("Warning: alpha_sequence is empty. Using default alpha=1.0 (balanced).")
        alpha_sequence = [1.0]
        drift_intervals = []

    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    extended_drift_intervals = []
    if drift_intervals:
        max_initial_drift_interval = drift_intervals[-1]
        num_drift_epochs = 10
        for i in range(num_drift_epochs):
            extended_drift_intervals.extend([interval + i * max_initial_drift_interval for interval in drift_intervals])
    else:
        extended_drift_intervals = [float('inf')]

    data_by_phase = []
    for phase_idx, alpha_val in enumerate(alpha_sequence):
        client_data_indices = [[] for _ in range(num_clients)]
        class_indices = [np.where(targets == i)[0] for i in range(num_classes)]

        for c in range(num_classes):
            dirichlet_alpha = alpha_val if isinstance(alpha_val, (int, float)) else alpha_val[c % len(alpha_val)]
            effective_dirichlet_alpha = max(dirichlet_alpha, 1e-4) * np.ones(num_clients)
            proportions = np.random.dirichlet(effective_dirichlet_alpha)
            num_samples_in_class = len(class_indices[c])
            proportions_samples = (proportions * num_samples_in_class).astype(float)
            proportions_int = np.floor(proportions_samples).astype(int)
            remainder = proportions_samples - proportions_int
            deficit = num_samples_in_class - proportions_int.sum()

            if deficit < 0:
                 indices_to_reduce = np.argsort(proportions_int)[::-1]
                 for i in range(abs(deficit)):
                     proportions_int[indices_to_reduce[i % num_clients]] -= 1
                 proportions_int = np.maximum(0, proportions_int)
                 deficit = 0

            indices_sorted_by_remainder = np.argsort(remainder)[::-1]
            for i in range(deficit):
                 proportions_int[indices_sorted_by_remainder[i % num_clients]] += 1

            if proportions_int.sum() != num_samples_in_class:
                final_diff = num_samples_in_class - proportions_int.sum()
                for i in range(abs(final_diff)):
                     proportions_int[i % num_clients] += np.sign(final_diff)
                proportions_int = np.maximum(0, proportions_int)

            indices = np.array(class_indices[c])
            np.random.shuffle(indices)
            current_sum = 0
            splits = []
            for p_val in proportions_int:
                splits.append(indices[current_sum : current_sum + p_val])
                current_sum += p_val

            for i, client_idx_list in enumerate(client_data_indices):
                if i < len(splits):
                    client_idx_list.extend(splits[i])
        data_by_phase.append(client_data_indices)

    def get_client_data_for_round(round_num):
        phase = 0
        if extended_drift_intervals and extended_drift_intervals[0] != float('inf'):
            phase = sum([round_num >= interval for interval in extended_drift_intervals]) % len(alpha_sequence)

        client_loaders = {}
        if phase >= len(data_by_phase):
             phase = 0
        current_phase_data_indices = data_by_phase[phase]

        for client_id, indices in enumerate(current_phase_data_indices):
            valid_indices = [idx for idx in indices if isinstance(idx, (int, np.integer))]
            if len(valid_indices) > 0:
                subset = Subset(dataset, valid_indices)
                if len(subset) == 0: continue
                batch_size = min(50, len(subset))
                if batch_size > 0:
                    client_loaders[client_id] = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
        return client_loaders, phase
    return get_client_data_for_round

# -----------------------------
# 2. DEFINIÇÃO DO MODELO (CNN Simples)
# -----------------------------
class CNN5(nn.Module):
    def __init__(self, num_classes=10): # Adicionado num_classes como parâmetro
        super(CNN5, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes) # Usando num_classes
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -----------------------------
# 3. UTILITÁRIOS DO FL
# -----------------------------
def train_client(model, dataloader, optimizer, criterion, frozen_layers=None, device='cpu'):
    model.train()
    if frozen_layers is None:
        frozen_layers = []

    original_requires_grad = {}
    for layer_name, layer in model.named_children():
        original_requires_grad[layer_name] = {}
        is_layer_explicitly_frozen = layer_name in frozen_layers
        for param_name, param in layer.named_parameters():
             original_requires_grad[layer_name][param_name] = param.requires_grad
             param.requires_grad = not is_layer_explicitly_frozen

    total_loss = 0.0
    total_samples = 0
    trainable_params_exist = any(p.requires_grad for p in model.parameters())

    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if trainable_params_exist:
             loss.backward()
             optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    for layer_name, layer in model.named_children():
        for param_name, param in layer.named_parameters():
            if layer_name in original_requires_grad and param_name in original_requires_grad[layer_name]:
                 param.requires_grad = original_requires_grad[layer_name][param_name]
            else:
                 param.requires_grad = True # Default para treinável se não rastreado
    if total_samples == 0: return 0.0
    return total_loss / total_samples

def aggregate_models(global_model, client_models):
    if not client_models: return
    global_state = global_model.state_dict()
    client_states = [client.state_dict() for client in client_models]
    for key in global_state.keys():
        if all(key in state for state in client_states):
            if client_states[0][key].dtype in [torch.float32, torch.float16, torch.float64]:
                global_state[key] = torch.stack(
                    [state[key].float() for state in client_states], dim=0
                ).mean(dim=0).to(global_state[key].dtype)
            else:
                global_state[key] = client_states[0][key]
    global_model.load_state_dict(global_state)

def evaluate_model(model, dataloader, criterion, device='cpu'):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item() * inputs.size(0)
    if total == 0: return 0.0, 0.0
    return 100.0 * correct / total, total_loss / total

# -----------------------------
# 4. CLASSE ALFADAPT (Com lógica proativa e regra para não congelar a última camada)
# -----------------------------
class ALFAdapt:
    def __init__(self, freeze_threshold=0.1, unfreeze_threshold=0.2, alpha_ema=0.95, last_layer_name='fc3'): # Adicionado last_layer_name
        self.freeze_threshold = freeze_threshold
        self.unfreeze_threshold = unfreeze_threshold
        self.alpha_ema = alpha_ema
        self.stability_monitor = {}
        self.layer_frozen_status = {}
        self.layer_frozen_at_round = {}
        self.verbose = False
        self.last_layer_name = last_layer_name # Nome da última camada que não deve ser congelada

    def initialize_monitor(self, model):
        if self.verbose: print("[ALFAdapt Init] Initializing stability monitor...")
        for name, param in model.named_parameters():
            layer_name = name.split(".")[0]
            self.stability_monitor[name] = {
                "ema_diff": torch.zeros_like(param.data, device=param.device),
                "ema_magnitude": torch.zeros_like(param.data, device=param.device),
                "stability_index": 1.0,
            }
            if layer_name not in self.layer_frozen_status:
                self.layer_frozen_status[layer_name] = False
                self.layer_frozen_at_round[layer_name] = -1
        if self.verbose:
             print("[ALFAdapt Init] Initial layer status:", self.layer_frozen_status)
        if self.verbose: print("[ALFAdapt Init] Monitor initialized.")

    def update_stability(self, global_model, previous_global_model_state):
        if self.verbose: print("  [ALFAdapt Update Stability] Updating parameter stability indices...")
        with torch.no_grad():
             for name, param in global_model.named_parameters():
                 if name not in previous_global_model_state or name not in self.stability_monitor:
                     if self.verbose: print(f"    Skipping stability update for {name} (not in previous state or monitor)")
                     continue
                 diff = param.data - previous_global_model_state[name]
                 abs_diff = diff.abs()
                 monitor = self.stability_monitor[name]
                 monitor["ema_diff"].mul_(self.alpha_ema).add_(diff, alpha=1 - self.alpha_ema)
                 monitor["ema_magnitude"].mul_(self.alpha_ema).add_(abs_diff, alpha=1 - self.alpha_ema)
                 numerator = monitor["ema_diff"].abs().mean()
                 denominator = monitor["ema_magnitude"].mean() + 1e-8
                 stability_value = (numerator / denominator).item()
                 monitor["stability_index"] = np.clip(stability_value, 0, 100.0)

    def freeze_unfreeze_layers(self, current_round):
        if self.verbose:
            print(f"[ALFAdapt Logic - Round {current_round}] Starting freeze/unfreeze decisions...")
            print(f"  Current Status Before Logic: {pprint.pformat(self.layer_frozen_status)}")

        changed_layers_this_call = {}
        layer_wise_stability = {}
        layer_param_counts = {}
        for name, monitor in self.stability_monitor.items():
            layer_name = name.split(".")[0]
            if layer_name not in layer_wise_stability:
                layer_wise_stability[layer_name] = 0.0
                layer_param_counts[layer_name] = 0
            layer_wise_stability[layer_name] += monitor["stability_index"]
            layer_param_counts[layer_name] += 1

        for layer_name in layer_wise_stability:
            if layer_param_counts.get(layer_name, 0) > 0:
                layer_wise_stability[layer_name] /= layer_param_counts[layer_name]
            else:
                layer_wise_stability[layer_name] = 1.0
        
        if self.verbose: print(f"  Calculated Layer Stabilities: {pprint.pformat({k: f'{v:.4f}' for k, v in layer_wise_stability.items()})}")

        for layer_name, current_stability_value in layer_wise_stability.items():
            if layer_name not in self.layer_frozen_status:
                 if self.verbose: print(f"    Warning: Layer {layer_name} not found in layer_frozen_status during decision making. Skipping.")
                 continue

            is_frozen_before_decision = self.layer_frozen_status[layer_name]
            
            if self.verbose:
                print(f"    Layer: {layer_name}, S_l: {current_stability_value:.4f}, Freeze_Thresh: {self.freeze_threshold:.4f}, Unfreeze_Thresh: {self.unfreeze_threshold:.4f}, Is_Currently_Frozen: {is_frozen_before_decision}")

            # --- MUDANÇA CHAVE: OPÇÃO A - NUNCA CONGELAR A ÚLTIMA CAMADA ---
            if layer_name == self.last_layer_name:
                if self.verbose: print(f"      Skipping freeze decision for output layer {layer_name}. It remains trainable.")
                if is_frozen_before_decision: # Se por acaso estava congelada, descongela
                    self.layer_frozen_status[layer_name] = False
                    changed_layers_this_call[layer_name] = f"UNFROZEN (Output Layer Policy)"
                    if self.verbose: print(f"      DEBUG: Output layer {layer_name} was frozen, now UNFROZEN by policy. Current changed_layers: {changed_layers_this_call}")
                continue # Pula para a próxima camada
            # --- FIM DA MUDANÇA CHAVE ---

            # Condição para congelar (outras camadas)
            if not is_frozen_before_decision and current_stability_value < self.freeze_threshold:
                if self.verbose: print(f"      --> Condition MET to FREEZE {layer_name}")
                self.layer_frozen_status[layer_name] = True
                self.layer_frozen_at_round[layer_name] = current_round
                changed_layers_this_call[layer_name] = f"FROZEN (S={current_stability_value:.4f})"
                if self.verbose: print(f"      DEBUG: Added to changed_layers: {layer_name} -> FROZEN. Current changed_layers: {changed_layers_this_call}")
            # Condição para descongelar (outras camadas)
            elif is_frozen_before_decision and current_stability_value > self.unfreeze_threshold:
                if self.verbose: print(f"      --> Condition MET to UNFREEZE {layer_name}")
                self.layer_frozen_status[layer_name] = False
                changed_layers_this_call[layer_name] = f"UNFROZEN (S={current_stability_value:.4f})"
                if self.verbose: print(f"      DEBUG: Added to changed_layers: {layer_name} -> UNFROZEN. Current changed_layers: {changed_layers_this_call}")
            else:
                if self.verbose: print(f"      Condition NOT met for layer {layer_name} to change status (S_l={current_stability_value:.4f}).")

        if self.verbose:
            print(f"  Status *AFTER ALL DECISIONS* in freeze_unfreeze_layers for Round {current_round}: {pprint.pformat(self.layer_frozen_status)}")
            if changed_layers_this_call:
                print(f"  ALF Logic Changes this round (final content of changed_layers_this_call): {pprint.pformat(changed_layers_this_call)}")
            else:
                print("  ALF Logic: No layer status changed based on stability (final content of changed_layers_this_call is empty).")

        frozen_layers_list = [lname for lname, frozen in self.layer_frozen_status.items() if frozen]
        return frozen_layers_list

# -----------------------------
# 5. LOOP PRINCIPAL DO FL (Com lógica proativa e verificações)
# -----------------------------
if __name__ == "__main__":
    NUM_CLIENTS = 100
    NUM_ROUNDS = 100 
    PARTICIPATION_RATE = 0.1
    BATCH_SIZE = 50
    LR = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    ALPHA_SEQUENCE = [0.3, 10.0, 0.1, 1.0, 0.2]
    DRIFT_INTERVALS = [20, 40, 60, 80]
    
    FREEZE_THRESHOLD = 0.25
    UNFREEZE_THRESHOLD = 0.35
    ALPHA_EMA_STABILITY = 0.9
    
    W_PERF = 5
    DELTA_PERF = 2.0
    K_PATIENCE = 3 
    N_UNFREEZE_PROACTIVE = 1
    ENABLE_PROACTIVE_UNFREEZING = True
    ALF_VERBOSE = True 
    GLOBAL_LOOP_VERBOSE = True
    
    # Nome da última camada do modelo CNN5 para a regra de não congelamento
    LAST_LAYER_TO_KEEP_TRAINABLE = 'fc3'


    print("Preparing data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
    get_client_data_loaders_for_round = create_dynamic_non_iid_data(
        train_dataset, NUM_CLIENTS, ALPHA_SEQUENCE, DRIFT_INTERVALS
    )
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    print("Data preparation finished.")

    print("Initializing model and ALFadapt...")
    global_model = CNN5(num_classes=10).to(device) # Passa num_classes
    criterion = nn.CrossEntropyLoss()
    # Passa o nome da última camada para o construtor do ALFAdapt
    alf = ALFAdapt(FREEZE_THRESHOLD, UNFREEZE_THRESHOLD, ALPHA_EMA_STABILITY, last_layer_name=LAST_LAYER_TO_KEEP_TRAINABLE)
    alf.verbose = ALF_VERBOSE
    alf.initialize_monitor(global_model)
    previous_global_state = {name: param.clone().detach().to(device) for name, param in global_model.state_dict().items()}

    log_records = []
    acc_history = deque(maxlen=W_PERF)
    acc_ma = 0.0
    rounds_since_degradation = 0
    active_phase = -1
    num_participants = int(NUM_CLIENTS * PARTICIPATION_RATE)

    print(f"Starting Federated Learning Loop for {NUM_ROUNDS} rounds...")
    for rnd in range(NUM_ROUNDS):
        current_round_num = rnd + 1
        if GLOBAL_LOOP_VERBOSE: print(f"\n--- Starting Round {current_round_num}/{NUM_ROUNDS} ---")

        client_data_map, current_phase = get_client_data_loaders_for_round(rnd)
        if active_phase != current_phase:
            if GLOBAL_LOOP_VERBOSE: print(f"** Concept Drift: Switched to Alpha Phase {current_phase} (Alpha={ALPHA_SEQUENCE[current_phase]}) **")
            active_phase = current_phase

        if not client_data_map:
            if GLOBAL_LOOP_VERBOSE: print(f"Warning: No client data for round {current_round_num}. Skipping.")
            log_records.append([current_round_num, 0, 0, acc_history[-1] if acc_history else 0, "SKIPPED_NO_DATA"])
            continue

        available_clients_with_data = list(client_data_map.keys())
        if not available_clients_with_data:
            if GLOBAL_LOOP_VERBOSE: print(f"Warning: No clients with data loaders for round {current_round_num}. Skipping.")
            log_records.append([current_round_num, 0, 0, acc_history[-1] if acc_history else 0, "SKIPPED_NO_CLIENT_LOADERS"])
            continue
            
        actual_participants_count = min(num_participants, len(available_clients_with_data))
        if actual_participants_count == 0:
            if GLOBAL_LOOP_VERBOSE: print(f"Warning: Not enough participants for round {current_round_num}. Skipping.")
            log_records.append([current_round_num, 0, 0, acc_history[-1] if acc_history else 0, "SKIPPED_NO_PARTICIPANTS"])
            continue

        selected_clients_ids = random.sample(available_clients_with_data, actual_participants_count)
        if GLOBAL_LOOP_VERBOSE: print(f"[Round {current_round_num}] Selected {len(selected_clients_ids)} clients for training.")

        client_models, local_losses = [], []
        current_frozen_layers_for_training = [lname for lname, fstatus in alf.layer_frozen_status.items() if fstatus]
        if GLOBAL_LOOP_VERBOSE: print(f"[Round {current_round_num}] START - Layers frozen for client training: {current_frozen_layers_for_training}")

        for client_id in selected_clients_ids:
            local_model = CNN5(num_classes=10).to(device) # Passa num_classes
            local_model.load_state_dict(global_model.state_dict())
            
            temp_frozen_layers_for_client = current_frozen_layers_for_training 
            for lname_iter, layer_iter in local_model.named_children():
                should_be_frozen = lname_iter in temp_frozen_layers_for_client
                for p_iter in layer_iter.parameters():
                    p_iter.requires_grad = not should_be_frozen
            
            list_optimizer_params = [p for p in local_model.parameters() if p.requires_grad]
            if not list_optimizer_params:
                 local_losses.append(0.0) 
                 # Adicionar o modelo não treinado pode ser importante se o ALFAdapt não for chamado se não houver agregação
                 client_models.append(local_model) # Adiciona modelo mesmo se não treinou, para que ALFAdapt seja chamado
                 if GLOBAL_LOOP_VERBOSE and ALF_VERBOSE: print(f"  Client {client_id} in Round {current_round_num}: No trainable parameters. Model added without training for ALF stability update.")
                 continue 

            optimizer = optim.SGD(list_optimizer_params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            
            local_loss = train_client(local_model, client_data_map[client_id], optimizer, criterion, temp_frozen_layers_for_client, device)
            local_losses.append(local_loss)
            client_models.append(local_model)

        aggregation_performed = False
        # Modificado para chamar ALF mesmo se apenas 1 modelo (ou modelos não treinados) forem adicionados,
        # para que a estabilidade possa ser calculada se o modelo global mudou devido a outros fatores (raro aqui) ou para manter o fluxo.
        if client_models: 
            if GLOBAL_LOOP_VERBOSE: print(f"[Round {current_round_num}] Aggregating {len(client_models)} client models...")
            aggregate_models(global_model, client_models) # Agrega mesmo se alguns não treinaram (foram adicionados como cópias)
            aggregation_performed = True # Considera-se que a agregação ocorreu para fins de atualização do ALF
        else: # Este caso não deve ocorrer se adicionarmos modelos mesmo sem treino
            if GLOBAL_LOOP_VERBOSE: print(f"[Round {current_round_num}] No client models to aggregate.")

        # Chamamos update_stability e freeze_unfreeze_layers mesmo que aggregation_performed seja False
        # se quisermos que o ALFAdapt continue avaliando o modelo global (por exemplo, se o modelo global pudesse mudar por outras razões)
        # Mas, no nosso caso, se não houve agregação real de modelos treinados, os pesos não mudaram.
        # No entanto, para manter o fluxo de ALF e permitir o descongelamento proativo mesmo que os pesos não mudem por um tempo,
        # vamos chamar alf.freeze_unfreeze_layers.
        # update_stability só faz sentido se previous_global_state for diferente do atual.
        
        if aggregation_performed : # Apenas atualiza estabilidade se houve agregação de modelos que podem ter mudado
            alf.update_stability(global_model, previous_global_state)
            previous_global_state = {name: param.clone().detach().to(device) for name, param in global_model.state_dict().items()}

        _ = alf.freeze_unfreeze_layers(current_round_num) # Chamamos isso toda rodada para aplicar a política da última camada, etc.
            
        test_acc, test_loss = evaluate_model(global_model, test_loader, criterion, device)
        current_accuracy = test_acc

        if ENABLE_PROACTIVE_UNFREEZING:
            acc_history.append(current_accuracy)
            valid_history = [acc for acc in acc_history if acc is not None]
            if len(valid_history) > 0:
                acc_ma = np.mean(valid_history)
                if GLOBAL_LOOP_VERBOSE:
                    print(f"[Round {current_round_num}] EVAL - Test Acc: {current_accuracy:.2f}%, Test Loss: {test_loss:.4f}")
                    print(f"  Performance History (last {W_PERF}): {[f'{a:.2f}' for a in acc_history]}, MA: {acc_ma:.2f}%")
            else:
                acc_ma = 0.0
                if GLOBAL_LOOP_VERBOSE:
                    print(f"[Round {current_round_num}] EVAL - Test Acc: {current_accuracy:.2f}%, Test Loss: {test_loss:.4f}")
                    print(f"  Performance History: Not enough valid data for MA yet.")
            
            degradation_detected_this_round = False
            if len(valid_history) >= W_PERF: 
                if current_accuracy < acc_ma - DELTA_PERF:
                    degradation_detected_this_round = True
                    rounds_since_degradation += 1
                    if GLOBAL_LOOP_VERBOSE: print(f"  [PROACTIVE CHECK {current_round_num}] ** Degradation DETECTED! ** Acc ({current_accuracy:.2f}) < MA ({acc_ma:.2f}) - Delta ({DELTA_PERF}). Rounds since deg: {rounds_since_degradation}")
                else: 
                    if rounds_since_degradation > 0 and GLOBAL_LOOP_VERBOSE: 
                         print(f"  [PROACTIVE CHECK {current_round_num}] Performance recovered or stable (Acc: {current_accuracy:.2f}%, MA: {acc_ma:.2f}%). Resetting degradation counter.")
                    rounds_since_degradation = 0 
            
            if degradation_detected_this_round and rounds_since_degradation > K_PATIENCE: 
                if GLOBAL_LOOP_VERBOSE: print(f"  [PROACTIVE ACTION {current_round_num}] --> Proactive Unfreeze TRIGGERED (Patience {K_PATIENCE} exceeded, Deg.Rounds: {rounds_since_degradation}) <--")
                candidate_frozen_layers = {lname: alf.layer_frozen_at_round.get(lname, -1) for lname, fstatus in alf.layer_frozen_status.items() if fstatus}
                if GLOBAL_LOOP_VERBOSE:
                    print(f"  [PROACTIVE ACTION {current_round_num}] Current frozen layers: {list(candidate_frozen_layers.keys())}")
                    if candidate_frozen_layers: print(f"  [PROACTIVE ACTION {current_round_num}] Frozen at rounds: {pprint.pformat({k:v for k,v in alf.layer_frozen_at_round.items() if k in candidate_frozen_layers})}")

                if candidate_frozen_layers:
                    sorted_candidates = sorted(candidate_frozen_layers.items(), key=lambda item: item[1], reverse=True)
                    if GLOBAL_LOOP_VERBOSE: print(f"  [PROACTIVE ACTION {current_round_num}] Candidates sorted by recency: {[(c[0], c[1]) for c in sorted_candidates]}")
                    unfrozen_count_this_step = 0
                    for i in range(min(N_UNFREEZE_PROACTIVE, len(sorted_candidates))):
                        layer_to_unfreeze, _ = sorted_candidates[i]
                        if alf.layer_frozen_status.get(layer_to_unfreeze, False): 
                            alf.layer_frozen_status[layer_to_unfreeze] = False 
                            unfrozen_count_this_step += 1
                            if GLOBAL_LOOP_VERBOSE: print(f"  [PROACTIVE ACTION {current_round_num}] ** Proactively unfroze layer '{layer_to_unfreeze}' for next round. **")
                    if unfrozen_count_this_step > 0:
                        rounds_since_degradation = 0 
                        if GLOBAL_LOOP_VERBOSE: print(f"  [PROACTIVE ACTION {current_round_num}] Degradation counter reset after unfreezing.")
                else:
                    if GLOBAL_LOOP_VERBOSE: print(f"  [PROACTIVE ACTION {current_round_num}] Triggered, but no frozen layers available to proactively unfreeze.")
        else: 
            if GLOBAL_LOOP_VERBOSE:
                print(f"[Round {current_round_num}] EVAL - Test Acc: {current_accuracy:.2f}%, Test Loss: {test_loss:.4f}")
                print(f"  Proactive Unfreezing Disabled.")

        avg_train_loss = sum(local_losses) / len(local_losses) if local_losses else 0.0
        final_frozen_for_log = [lname for lname, fstatus in alf.layer_frozen_status.items() if fstatus]
        if GLOBAL_LOOP_VERBOSE: print(f"[Round {current_round_num}] END - Frozen layers for NEXT round: {final_frozen_for_log}")
        log_records.append([current_round_num, avg_train_loss, test_loss, current_accuracy, ",".join(final_frozen_for_log) if final_frozen_for_log else "None"])

    log_filename = f"federated_log_proactive_FT_{FREEZE_THRESHOLD}_UT_{UNFREEZE_THRESHOLD}_KP_{K_PATIENCE}.csv"
    print(f"\nTraining finished. Saving logs to {log_filename}...")
    try:
        with open(log_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "AvgTrainLoss", "TestLoss", "TestAccuracy", "FrozenLayersForNextRound"])
            for row in log_records:
                writer.writerow(row)
        print("Logs saved successfully.")
    except Exception as e:
        print(f"Error saving logs: {e}")
    print("===== Experiment Finished =====")