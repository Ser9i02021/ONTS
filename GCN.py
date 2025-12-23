import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool


# ============================================================
# Utilidades gerais
# ============================================================

def set_seed(seed: int = 0) -> None:
    """Define sementes para melhorar reprodutibilidade (não garante 100% determinismo)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_to_device(data: Data, device: torch.device) -> Data:
    """Move um objeto PyG Data para o device (CPU/GPU)."""
    return data.to(device)


# ============================================================
# Rede Neural Gráfica (GNN) para aproximar Q(s, a)
# ============================================================

class GNN(torch.nn.Module):
    """
    Q-network baseado em GCN + pooling global.
    Entrada: grafo (nós = job×tempo, features por nó).
    Saída: vetor Q com dimensão = número de ações (J*T).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Camadas de convolução em grafos (GCNConv)
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 64)

        # Cabeça MLP (fully-connected)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass:
        - Convoluções GCN
        - Pooling global mean
        - MLP para produzir Q-values por ação
        """
        x, edge_index = data.x, data.edge_index

        # Em Batch graphs, data.batch existe; em um grafo único (Data), não.
        # Se não existir, criamos batch=0 para todos os nós.
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 1ª camada GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 2ª camada GCN
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 3ª camada GCN
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Pooling global por grafo
        x = global_mean_pool(x, batch)

        # MLP final
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


# ============================================================
# Ambiente ONTS (Scheduling)
# ============================================================

class ONTSEnv:
    """
    Ambiente para problema ONTS (jobs × time steps).
    Ação: selecionar (job, time_step) e alternar x[j,t] entre 0/1.
    """

    def __init__(
        self,
        u__job_priorities,
        q__energy_consumption_per_job,
        y_min_per_job,
        y_max_per_job,
        t_min_per_job,
        t_max_per_job,
        p_min_per_job,
        p_max_per_job,
        w_min_per_job,
        w_max_per_job,
        r__energy_available_at_time_t,
        gamma,
        Vb,
        Q,
        p,
        e,
        max_steps=None,
        add_time_coupling_edges: bool = True,  # Edge extra: acoplar jobs no mesmo time-step
    ):
        # Parâmetros do problema / restrições
        self.u__job_priorities = np.array(u__job_priorities)
        self.q__energy_consumption_per_job = np.array(q__energy_consumption_per_job)

        self.y_min_per_job = y_min_per_job
        self.y_max_per_job = y_max_per_job

        self.t_min_per_job = t_min_per_job
        self.t_max_per_job = t_max_per_job

        self.p_min_per_job = p_min_per_job
        self.p_max_per_job = p_max_per_job

        self.w_min_per_job = w_min_per_job
        self.w_max_per_job = w_max_per_job

        self.r__energy_available_at_time_t = np.array(r__energy_available_at_time_t)

        # Parâmetros bateria/energia
        self.gamma = gamma
        self.Vb = Vb
        self.Q = Q
        self.p = p
        self.e = e

        # Estado de carga (SoC)
        self.SoC_t = self.p

        # Dimensões
        self.J = len(self.u__job_priorities)
        self.T = len(self.r__energy_available_at_time_t)

        # Limite de passos por episódio
        self.max_steps = max_steps if max_steps is not None else self.T

        # Flags para grafo
        self.add_time_coupling_edges = add_time_coupling_edges

        # Estado interno
        self.x__state = None
        self.phi__state = None
        self.steps_taken = 0

        # Inicializa
        self.reset()

    def reset(self):
        """Reinicia o ambiente para o estado inicial."""
        self.x__state = np.zeros((self.J, self.T), dtype=int)
        self.phi__state = np.zeros((self.J, self.T), dtype=int)
        self.steps_taken = 0
        self.SoC_t = self.p
        return self.x__state.flatten()

    def step(self, action: int):
        """
        Executa uma ação (toggle em x[j,t]), atualiza o ambiente, e retorna:
        next_state_matrix, reward, done
        """
        job, time_step = divmod(action, self.T)
        self.steps_taken += 1

        # Toggle da decisão
        self.x__state[job, time_step] = 1 - self.x__state[job, time_step]

        # Atualiza phi (matriz auxiliar)
        self.build_phi_matrix()

        # Calcula recompensa e se houve violação "terminal"
        reward, violated = self.calculate_reward()

        # Episódio termina se violou restrição "terminal" ou excedeu max_steps
        done = violated or (self.steps_taken >= self.max_steps)

        # Retornamos uma cópia para evitar mutações externas inesperadas
        return self.x__state.copy(), reward, done

    def build_phi_matrix(self):
        """Constrói/atualiza a matriz phi para acompanhar ativações/desativações."""
        for j in range(self.J):
            for t in range(self.T):
                if t == 0:
                    if self.x__state[j, t] > self.phi__state[j, t]:
                        self.phi__state[j, t] = 1
                else:
                    # Marca subida (ativação) quando x(t)-x(t-1) = 1
                    if (self.x__state[j, t] - self.x__state[j, t - 1]) > self.phi__state[j, t]:
                        self.phi__state[j, t] = 1

                    # Corrige casos inválidos
                    if self.phi__state[j, t] > (2 - self.x__state[j, t] - self.x__state[j, t - 1]):
                        self.phi__state[j, t] = 0

                # phi não pode ser 1 se x é 0
                if self.phi__state[j, t] > self.x__state[j, t]:
                    self.phi__state[j, t] = 0

    def check_energy_constraints(self):
        """
        Verifica restrições de energia e atualiza SoC.
        Retorna (reward, done_violation).
        - done_violation=True quando há violação "terminal".
        """
        for t in range(self.T):
            total_energy_required = 0

            # Soma consumo total no time-step t
            for j in range(self.J):
                total_energy_required += self.x__state[j, t] * self.q__energy_consumption_per_job[j]

            # Se excede energia disponível + margem bateria => violação terminal
            if total_energy_required > self.r__energy_available_at_time_t[t] + (self.gamma * self.Vb):
                return -1, True

            # Atualiza SoC (modelo simples)
            exceeding_power = self.r__energy_available_at_time_t[t] - total_energy_required
            i_t = exceeding_power / self.Vb
            self.SoC_t = self.SoC_t + (i_t * self.e) / (60 * self.Q)

            # Se excede SoC máximo => violação terminal
            if self.SoC_t > 1:
                return -1, True

        # Sem violação
        return 0, False

    def check_job_constraints(self):
        """Verifica restrições de cada job. Retorna penalização acumulada (<=0)."""
        acc_reward = 0

        for j in range(self.J):
            # Janela permitida (w_min, w_max)
            for tw in range(self.w_min_per_job[j]):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1

            for tw in range(self.w_max_per_job[j], self.T):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1

            # Restrição y_min / y_max baseada em phi
            sum_phi = 0
            for t in range(self.T):
                sum_phi += self.phi__state[j, t]

            if sum_phi < self.y_min_per_job[j]:
                acc_reward -= 1
            if sum_phi > self.y_max_per_job[j]:
                acc_reward -= 1

            # Execução contínua mínima (t_min)
            for t in range(self.T - self.t_min_per_job[j] + 1):
                tt_sum = 0
                for tt in range(t, t + self.t_min_per_job[j]):
                    tt_sum += self.x__state[j, tt]
                if tt_sum < self.t_min_per_job[j] * self.phi__state[j, t]:
                    acc_reward -= 1

            # Execução contínua máxima (t_max)
            for t in range(self.T - self.t_max_per_job[j]):
                tt_sum = 0
                for tt in range(t, t + self.t_max_per_job[j] + 1):
                    tt_sum += self.x__state[j, tt]
                if tt_sum > self.t_max_per_job[j]:
                    acc_reward -= 1

            # Periodicidade mínima (p_min): não pode ter >1 ativação no intervalo
            for t in range(self.T - self.p_min_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_min_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l > 1:
                    acc_reward -= 1

            # Periodicidade máxima (p_max): deve ter pelo menos 1 ativação no intervalo
            for t in range(self.T - self.p_max_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_max_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l < 1:
                    acc_reward -= 1

        return acc_reward

    def calculate_reward(self):
        """
        Calcula recompensa total.
        - Primeiro verifica energia (pode ser terminal).
        - Depois verifica restrições de job (penalizações).
        - Se rewardSum==0, adiciona um termo positivo guiado por prioridades e folga de energia.
        """
        reward_sum = 0

        # 1) Energia primeiro (terminal se violar)
        reward_energy, violated = self.check_energy_constraints()
        reward_sum += reward_energy

        if violated:
            return reward_sum, True

        # 2) Restrições do job
        reward_jobs = self.check_job_constraints()
        reward_sum += reward_jobs

        # 3) Recompensa positiva somente se não houve penalização alguma
        if reward_sum == 0:
            for j in range(self.J):
                for t in range(self.T):
                    if self.x__state[j, t] == 1:
                        # Prioridade * folga de energia (exemplo simplificado)
                        slack = self.r__energy_available_at_time_t[t] - self.q__energy_consumption_per_job[j]
                        reward_sum += 10.0 * self.u__job_priorities[j] * slack

        return reward_sum, False

    def create_edges(self):
        """
        Cria arestas do grafo:
        - Conecta tempos consecutivos dentro do mesmo job (cadeia).
        - (Opcional) Conecta jobs no mesmo time-step (acoplamento de recurso).
        """
        edges = []

        # Arestas temporais por job (bidirecionais)
        for job in range(self.J):
            for t in range(self.T - 1):
                a = job * self.T + t
                b = job * self.T + t + 1
                edges.append((a, b))
                edges.append((b, a))

        # Arestas de acoplamento no mesmo time-step entre jobs (cadeia job-job)
        # Isso ajuda a GNN a capturar o fato de que jobs competem pelo recurso energia no mesmo t.
        if self.add_time_coupling_edges and self.J > 1:
            for t in range(self.T):
                for job in range(self.J - 1):
                    a = job * self.T + t
                    b = (job + 1) * self.T + t
                    edges.append((a, b))
                    edges.append((b, a))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def get_graph(self):
        """
        Constrói um grafo PyG Data a partir do estado atual.

        Features por nó (exemplo enriquecido):
        - x_state[j,t] (0/1)
        - prioridade normalizada do job
        - consumo de energia normalizado do job
        - energia disponível normalizada no time-step
        """
        edge_index = self.create_edges()

        # Normalizações simples para manter features em escala razoável
        max_u = max(1.0, float(np.max(self.u__job_priorities)))
        max_q = max(1.0, float(np.max(self.q__energy_consumption_per_job)))
        max_r = max(1.0, float(np.max(self.r__energy_available_at_time_t)))

        node_features = []
        for j in range(self.J):
            for t in range(self.T):
                node_features.append([
                    float(self.x__state[j, t]),
                    float(self.u__job_priorities[j] / max_u),
                    float(self.q__energy_consumption_per_job[j] / max_q),
                    float(self.r__energy_available_at_time_t[t] / max_r),
                ])

        x = torch.tensor(node_features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index)


# ============================================================
# Replay buffer / Experience tuple
# ============================================================

# Agora incluímos done (terminal) para TD target correto.
Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))


# ============================================================
# Política ε-greedy
# ============================================================

def select_action_gnn(env: ONTSEnv, policy_net: GNN, epsilon: float, device: torch.device) -> int:
    """
    Seleção de ação ε-greedy:
    - com prob (1-ε) escolhe argmax Q
    - com prob ε escolhe aleatória
    """
    if random.random() > epsilon:
        with torch.no_grad():
            state_graph = data_to_device(env.get_graph(), device)
            q_values = policy_net(state_graph)          # shape: [1, n_actions]
            action = int(q_values.argmax(dim=1).item()) # pega a melhor ação
            return action
    else:
        return random.randrange(env.J * env.T)


# ============================================================
# Otimização: Double DQN + Target Network
# ============================================================

def optimize_model_gnn(
    policy_net: GNN,
    target_net: GNN,
    optimizer: optim.Optimizer,
    memory: deque,
    gamma: float,
    batch_size: int,
    device: torch.device,
    grad_clip: float = 1.0,
):
    """
    Atualiza policy_net usando:
    - Experience replay
    - Double DQN target
    - Huber loss
    """
    # Só treina quando há amostras suficientes
    if len(memory) < batch_size:
        return

    # Amostra batch aleatório do replay
    experiences = random.sample(memory, batch_size)
    batch = Experience(*zip(*experiences))

    # Batch de grafos (PyG) e tensores
    state_batch = Batch.from_data_list(list(batch.state)).to(device)
    next_state_batch = Batch.from_data_list(list(batch.next_state)).to(device)

    action_batch = torch.cat(batch.action).to(device)       # shape: [B, 1]
    reward_batch = torch.cat(batch.reward).to(device)       # shape: [B]
    done_batch = torch.cat(batch.done).to(device)           # shape: [B] com 0/1

    # Q(s,a) do policy_net para ações escolhidas
    q_sa = policy_net(state_batch).gather(1, action_batch).squeeze(1)  # shape: [B]

    # Double DQN:
    # 1) policy escolhe a* = argmax_a Q_policy(s',a)
    with torch.no_grad():
        next_actions = policy_net(next_state_batch).argmax(dim=1, keepdim=True)  # shape: [B, 1]

        # 2) target avalia Q_target(s', a*)
        q_next = target_net(next_state_batch).gather(1, next_actions).squeeze(1)  # shape: [B]

        # TD target: r + (1-done)*gamma*q_next
        target = reward_batch + (1.0 - done_batch) * gamma * q_next

    # Loss Huber (mais robusta que MSE)
    loss = F.smooth_l1_loss(q_sa, target)

    # Backprop
    optimizer.zero_grad()
    loss.backward()

    # (Opcional) clipping de gradiente para estabilidade
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)

    optimizer.step()


# ============================================================
# Treinamento com Target Net + Double DQN
# ============================================================

def train_gnn(
    env: ONTSEnv,
    episodes: int = 500,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay: float = 0.995,
    batch_size: int = 128,
    target_update_every: int = 200,  # atualiza target a cada N otimizações
    learning_rate: float = 1e-3,
    seed: int = 0,
):
    """
    Loop de treinamento:
    - Cria policy_net e target_net
    - Faz replay + Double DQN
    - Sincroniza target_net periodicamente
    """
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_actions = env.J * env.T

    # Agora temos 4 features por nó (ver get_graph)
    in_channels = 4

    # Policy e Target networks
    policy_net = GNN(in_channels=in_channels, out_channels=n_actions).to(device)
    target_net = GNN(in_channels=in_channels, out_channels=n_actions).to(device)

    # Inicialmente target = policy
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=10000)

    epsilon = eps_start
    num_optim_steps = 0
    episode_returns = []

    for episode in range(episodes):
        env.reset()
        total_reward = 0.0
        done = False

        # Cada episódio pode ter até max_steps, mas também pode terminar por violação
        while not done:
            # Captura estado ANTES do step (fix importante para replay)
            state_graph = env.get_graph()

            # Atualiza epsilon
            epsilon = max(epsilon * eps_decay, eps_end)

            # Seleciona ação
            action = select_action_gnn(env, policy_net, epsilon, device)

            # Executa ação
            _, reward, done = env.step(action)
            total_reward += float(reward)

            # Captura next_state DEPOIS do step
            next_state_graph = env.get_graph()

            # Armazena transição no replay buffer
            memory.append(
                Experience(
                    state=state_graph,
                    action=torch.tensor([[action]], dtype=torch.long),
                    reward=torch.tensor([reward], dtype=torch.float),
                    next_state=next_state_graph,
                    done=torch.tensor([1.0 if done else 0.0], dtype=torch.float),
                )
            )

            # Otimiza (Double DQN + Target Net)
            optimize_model_gnn(
                policy_net=policy_net,
                target_net=target_net,
                optimizer=optimizer,
                memory=memory,
                gamma=gamma,
                batch_size=batch_size,
                device=device,
            )
            num_optim_steps += 1

            # Atualiza target periodicamente
            if num_optim_steps % target_update_every == 0:
                target_net.load_state_dict(policy_net.state_dict())
                target_net.eval()

        episode_returns.append(total_reward)

    return policy_net, target_net, memory, episode_returns


# ============================================================
# Avaliação
# ============================================================

@torch.no_grad()
def evaluate_gnn_model(env: ONTSEnv, policy_net: GNN, episodes: int = 10):
    """
    Avalia a policy em modo guloso (epsilon=0).
    Retorna recompensa média por episódio e imprime schedules finais.
    """
    device = next(policy_net.parameters()).device

    total_returns = 0.0

    for ep in range(episodes):
        env.reset()
        done = False
        ep_return = 0.0

        while not done:
            # Epsilon=0 => greedy
            action = select_action_gnn(env, policy_net, epsilon=0.0, device=device)
            state, reward, done = env.step(action)
            ep_return += float(reward)

        total_returns += ep_return

        print(f"Episode {ep + 1}: Return = {ep_return}")
        print(state)
        print()

    avg_return = total_returns / episodes
    print(f"Average Return over {episodes} episodes: {avg_return}")
    return avg_return


# ============================================================
# Exemplo de uso (mesmo setup do seu script)
# ============================================================

if __name__ == "__main__":
    # Instância de teste pequena (3 jobs × 5 time steps)
    u__job_priorities = np.array([3, 2, 1])
    q__energy_consumption_per_job = np.array([1, 2, 1])

    y_min_per_job = [1, 1, 1]
    y_max_per_job = [3, 4, 5]

    t_min_per_job = [1, 1, 1]
    t_max_per_job = [3, 4, 3]

    p_min_per_job = [1, 1, 1]
    p_max_per_job = [4, 5, 5]

    w_min_per_job = [0, 0, 0]
    w_max_per_job = [4, 5, 4]

    r__energy_available_at_time_t = np.array([3, 3, 3, 3, 3])

    gamma = 0.5
    Vb = 1
    Q = 10
    p = 0.1
    e = 0.9

    env = ONTSEnv(
        u__job_priorities=u__job_priorities,
        q__energy_consumption_per_job=q__energy_consumption_per_job,
        y_min_per_job=y_min_per_job,
        y_max_per_job=y_max_per_job,
        t_min_per_job=t_min_per_job,
        t_max_per_job=t_max_per_job,
        p_min_per_job=p_min_per_job,
        p_max_per_job=p_max_per_job,
        w_min_per_job=w_min_per_job,
        w_max_per_job=w_max_per_job,
        r__energy_available_at_time_t=r__energy_available_at_time_t,
        gamma=gamma,
        Vb=Vb,
        Q=Q,
        p=p,
        e=e,
        add_time_coupling_edges=True,
    )

    # Treina várias vezes (como no seu exemplo) e faz média da avaliação
    runs = 10
    avg_sum = 0.0

    for run in range(runs):
        print(f"=== Run {run + 1}/{runs} ===")

        policy_net, target_net, memory, returns = train_gnn(
            env,
            episodes=2000,
            gamma=0.99,
            eps_start=1.0,
            eps_end=0.01,
            eps_decay=0.995,
            batch_size=128,
            target_update_every=200,
            learning_rate=1e-3,
            seed=run,  # muda a seed por run
        )

        avg_rew = evaluate_gnn_model(env, policy_net, episodes=10)
        avg_sum += avg_rew

    print()
    print(f"Final average over {runs} runs: {avg_sum / runs}")
