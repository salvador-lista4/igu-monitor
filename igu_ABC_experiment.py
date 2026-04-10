# ============================================================
# IGU — Experimento comparativo: A vs B vs C
# Agente A: gradiente puro (sin historia)
# Agente B: historia estructural (deforma V)  ← IGU
# Agente C: memoria explícita (buffer de estados, no deforma V)
#
# Propósito: demostrar que B y C son mecanismos cualitativamente
# distintos, respondiendo al revisor que solicita este control.
#
# Listo para Google Colab — pegar y ejecutar directamente
# ============================================================

import numpy as np
from scipy.linalg import eigh
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# PARÁMETROS
# ─────────────────────────────────────────────────────────────
AT   = np.array([[-2.,-2.],[2.,-2.],[-2.,2.],[2.,2.]])
WA   = 4.0;  WS = 0.75;  BC = 0.015;  GT = 0.10
RH   = 2.5;  SH = 1.2;   MU = 0.02
EA   = 0.12; EB = 0.05;  XI = 0.12
EPC  = 0.08; GC = 0.20
T    = 12.0; DT = 0.01;  NOISE = 0.03
N    = 200   # corridas por condición

# Agente C — parámetros de memoria explícita
ALPHA_C = 0.20   # fuerza de atracción hacia memoria
ETA_C   = 0.08   # movilidad base de C
MU_C    = 0.02   # decaimiento temporal (mismo que B para comparación justa)

# ─────────────────────────────────────────────────────────────
# MATEMÁTICA BASE
# ─────────────────────────────────────────────────────────────
def Vb(x, t3=False):
    v = sum(-WA*np.exp(-np.dot(x-a,x-a)/(2*WS**2)) for a in AT) + BC*np.dot(x,x)**2
    if t3: v += GT*(x[0]**3 - 3*x[0]*x[1]**2)
    return float(v)

def gVb(x, t3=False):
    g = sum(WA*np.exp(-np.dot(x-a,x-a)/(2*WS**2))*(x-a)/WS**2 for a in AT) + 4*BC*np.dot(x,x)*x
    if t3: g = g + GT*np.array([3*x[0]**2-3*x[1]**2, -6*x[0]*x[1]])
    return np.array(g, dtype=float)

def hVb(x, t3=False):
    H = np.zeros((2,2))
    for a in AT:
        d=x-a; r2=np.dot(d,d); e=np.exp(-r2/(2*WS**2))
        H += WA*e*(np.eye(2)/WS**2 - np.outer(d,d)/WS**4)
    H += 4*BC*(np.dot(x,x)*np.eye(2) + 2*np.outer(x,x))
    if t3: H += GT*np.array([[6*x[0],-6*x[1]],[-6*x[1],-6*x[0]]])
    return H

# ─────────────────────────────────────────────────────────────
# HISTORIA ESTRUCTURAL (Agente B) — deforma V
# ─────────────────────────────────────────────────────────────
class StructuralHistory:
    """Historia que deforma el paisaje V — mecanismo IGU."""
    def __init__(self, rho=RH, sigma=SH, mu=MU):
        self.rho=rho; self.sigma=sigma; self.mu=mu
        self.ts=[]; self.xs=[]

    def rec(self, x, t): self.xs.append(x.copy()); self.ts.append(t)

    def _idx(self, mp=80):
        return np.linspace(0,len(self.xs)-1,min(mp,len(self.xs)),dtype=int) if self.xs else []

    def _dte(self):
        return (self.ts[-1]-self.ts[0])/max(len(self.xs),1) if len(self.ts)>1 else 1.

    def gdV(self, x, t):
        """Gradiente de la deformación histórica de V."""
        g = np.zeros(2)
        for i in self._idx():
            d=x-self.xs[i]; r2=np.dot(d,d)
            e=np.exp(-r2/(2*self.sigma**2))*np.exp(-self.mu*max(t-self.ts[i],0.))
            g += self.rho*e*d/self.sigma**2
        return g*self._dte()

    def hdV(self, x, t):
        """Hessiano de la deformación histórica de V."""
        H = np.zeros((2,2))
        for i in self._idx():
            d=x-self.xs[i]; r2=np.dot(d,d)
            e=np.exp(-r2/(2*self.sigma**2))*np.exp(-self.mu*max(t-self.ts[i],0.))
            H += self.rho*e*(np.eye(2)/self.sigma**2 - np.outer(d,d)/self.sigma**4)
        return H*self._dte()

# ─────────────────────────────────────────────────────────────
# MEMORIA EXPLÍCITA (Agente C) — buffer de estados, NO deforma V
# ─────────────────────────────────────────────────────────────
class ExplicitMemory:
    """
    Memoria explícita: buffer de estados pasados con decaimiento temporal.
    NO modifica V. Genera una fuerza directa hacia estados recordados.
    Diferencia clave vs. StructuralHistory:
    - En StructuralHistory: la historia MODIFICA la geometría del paisaje
    - En ExplicitMemory: la historia ATRAE el agente hacia posiciones pasadas
      pero el paisaje V_base permanece sin cambios
    """
    def __init__(self, mu=MU_C):
        self.mu=mu; self.ts=[]; self.xs=[]

    def rec(self, x, t): self.xs.append(x.copy()); self.ts.append(t)

    def memory_force(self, x, t, alpha=ALPHA_C, mp=80):
        """
        Fuerza de atracción hacia estados recordados:
        f_mem(x,t) = alpha * Σ_i w_i * (x_mem_i - x)
        donde w_i = exp(-mu*(t-t_i)) / Z

        Esto es cualitativamente distinto de deformar V:
        - Aquí la fuerza apunta HACIA posiciones pasadas
        - En B la deformación PROFUNDIZA pozos en posiciones pasadas
        """
        if not self.xs: return np.zeros(2)
        idx = np.linspace(0,len(self.xs)-1,min(mp,len(self.xs)),dtype=int)
        weights = []
        forces  = []
        for i in idx:
            w = np.exp(-self.mu*max(t-self.ts[i], 0.))
            weights.append(w)
            forces.append(w * (self.xs[i] - x))
        Z = sum(weights) + 1e-12
        return alpha * sum(forces) / Z

    def is_recoverable(self):
        """
        La memoria explícita es recuperable: borrar el buffer
        restaura el sistema al estado sin historia.
        (Propiedad que StructuralHistory NO tiene)
        """
        return True

    def clear(self):
        """Borrar el buffer — imposible en StructuralHistory sin costo dinámico."""
        self.ts=[]; self.xs=[]

# ─────────────────────────────────────────────────────────────
# AGENTES
# ─────────────────────────────────────────────────────────────
def near(x): return int(np.argmin([np.linalg.norm(x-a) for a in AT]))

def runA(x0):
    """Agente A — gradiente puro, sin historia, sin memoria."""
    x=x0.copy(); lm=[]; cr=[]
    for _ in range(int(T/DT)):
        H=hVb(x); ev=eigh(H,eigvals_only=True); lmin=ev[0]
        g=gVb(x); gn=np.linalg.norm(g)
        lm.append(lmin); cr.append(abs(lmin)<EPC and gn<GC)
        x = x - DT*EA*g
    return near(x), np.mean(lm), float(sum(cr))*DT

def runB(x0, hist):
    """Agente B — historia estructural: deforma V."""
    x=x0.copy(); t=0.; lm=[]; cr=[]
    for _ in range(int(T/DT)):
        H=hVb(x,True)+hist.hdV(x,t)
        ev,evec=eigh(H); lmin=ev[0]
        g=gVb(x,True)+hist.gdV(x,t); gn=np.linalg.norm(g)
        lm.append(lmin); cr.append(abs(lmin)<EPC and gn<GC)
        Hp=evec@np.diag(np.maximum(ev,0.))@evec.T
        M=EB*np.eye(2)+XI*Hp
        x = x-DT*(M@g); t+=DT; hist.rec(x,t)
    return near(x), np.mean(lm), float(sum(cr))*DT

def runC(x0, mem):
    """
    Agente C — memoria explícita: buffer de estados con decaimiento.
    V_base NO se deforma. La memoria genera una fuerza directa.
    """
    x=x0.copy(); t=0.; lm=[]; cr=[]
    for _ in range(int(T/DT)):
        # El Hessiano es solo el del potencial base — V no cambia
        H=hVb(x,False)
        ev=eigh(H,eigvals_only=True); lmin=ev[0]
        g=gVb(x,False)
        gn_base=np.linalg.norm(g)
        lm.append(lmin); cr.append(abs(lmin)<EPC and gn_base<GC)

        # Fuerza de memoria: atracción directa hacia estados pasados
        f_mem = mem.memory_force(x, t)

        # Ecuación de movimiento: gradiente base + fuerza de memoria
        # Nota: la fuerza de memoria NO modifica V, es un término adicional
        x = x - DT*(ETA_C*g - f_mem)
        t += DT; mem.rec(x, t)
    return near(x), np.mean(lm), float(sum(cr))*DT

# ─────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE HISTORIAS
# ─────────────────────────────────────────────────────────────
def build_structural_history(wps):
    """Historia estructural para Agente B."""
    h=StructuralHistory(); x=np.array(wps[0],float); t=0.; h.rec(x,t)
    for wp in wps[1:]:
        wp=np.array(wp,float)
        for _ in range(int(3./DT)):
            d=wp-x; n=np.linalg.norm(d)
            x=x+DT*(0.8*d/n if n>1e-8 else np.zeros(2)); t+=DT; h.rec(x,t)
    return h

def build_explicit_memory(wps):
    """
    Memoria explícita para Agente C.
    Misma trayectoria que B — comparación justa.
    """
    m=ExplicitMemory(); x=np.array(wps[0],float); t=0.; m.rec(x,t)
    for wp in wps[1:]:
        wp=np.array(wp,float)
        for _ in range(int(3./DT)):
            d=wp-x; n=np.linalg.norm(d)
            x=x+DT*(0.8*d/n if n>1e-8 else np.zeros(2)); t+=DT; m.rec(x,t)
    return m

H1_WPS = [[-3.2,0.],[-2.3,-2.1],[0.,-3.],[2.3,-2.1]]   # inferior/conservador
H2_WPS = [[-3.2,0.],[-2.3, 2.1],[0., 3.],[2.3, 2.1]]   # superior/innovador

# ─────────────────────────────────────────────────────────────
# BARRIDO N=200
# ─────────────────────────────────────────────────────────────
print(f"Barrido comparativo A vs B vs C  (N={N})\n")
print("Agente A: gradiente puro")
print("Agente B: historia estructural (deforma V) — IGU")
print("Agente C: memoria explícita (buffer, no deforma V) — control\n")

cA  = np.zeros(4,int); cB1 = np.zeros(4,int); cB2 = np.zeros(4,int)
cC1 = np.zeros(4,int); cC2 = np.zeros(4,int)
lmA=[]; lmB1=[]; lmC1=[]
tcA=[]; tcB1=[]; tcC1=[]
div_BC1=[]; div_BC2=[]   # B vs C con misma historia

t0=time.time()
for run in range(N):
    x0 = np.array([0.,0.]) + np.random.normal(0,NOISE,2)

    aA,  lA,  tA  = runA(x0.copy())
    aB1, lB1, tB1 = runB(x0.copy(), build_structural_history(H1_WPS))
    aB2, lB2, tB2 = runB(x0.copy(), build_structural_history(H2_WPS))
    aC1, lC1, tC1 = runC(x0.copy(), build_explicit_memory(H1_WPS))
    aC2, lC2, tC2 = runC(x0.copy(), build_explicit_memory(H2_WPS))

    cA[aA]+=1; cB1[aB1]+=1; cB2[aB2]+=1
    cC1[aC1]+=1; cC2[aC2]+=1
    lmA.append(lA); lmB1.append(lB1); lmC1.append(lC1)
    tcA.append(tA); tcB1.append(tB1); tcC1.append(tC1)
    div_BC1.append(int(aB1!=aC1))   # ¿B y C difieren con la MISMA historia?
    div_BC2.append(int(aB2!=aC2))

    if (run+1) % 50 == 0:
        print(f"  {run+1}/{N}  —  {time.time()-t0:.0f}s")

print(f"\nCompletado en {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────
def entropy(c):
    p=c/c.sum(); p=p[p>0]; return float(-np.sum(p*np.log(p)))

div_B1B2 = sum(1 for i in range(N) if
    runB(np.array([0.,0.])+np.random.normal(0,NOISE,2),
         build_structural_history(H1_WPS))[0] !=
    runB(np.array([0.,0.])+np.random.normal(0,NOISE,2),
         build_structural_history(H2_WPS))[0]) / N * 100

div_C1C2_rate = sum(1 for i in range(N) if
    runC(np.array([0.,0.])+np.random.normal(0,NOISE,2),
         build_explicit_memory(H1_WPS))[0] !=
    runC(np.array([0.,0.])+np.random.normal(0,NOISE,2),
         build_explicit_memory(H2_WPS))[0]) / N * 100

div_BC1_rate = np.mean(div_BC1)*100
div_BC2_rate = np.mean(div_BC2)*100

# ─────────────────────────────────────────────────────────────
# TABLA
# ─────────────────────────────────────────────────────────────
print("\n" + "="*72)
print(f"TABLA COMPARATIVA  A vs B vs C  (N={N})")
print("="*72)
print(f"{'Agente':>12}  {'a0':>5}{'a1':>5}{'a2':>5}{'a3':>5}  {'Entr':>6}  {'λ_min':>8}  {'τ_c(s)':>8}")
print("-"*72)
for label, c, lm, tc in [
    ("A (gradiente)", cA,  lmA, tcA),
    ("B/H1 (IGU)",   cB1, lmB1, tcB1),
    ("C/H1 (buffer)", cC1, lmC1, tcC1),
]:
    print(f"{label:>12}  {c[0]:>5}{c[1]:>5}{c[2]:>5}{c[3]:>5}"
          f"  {entropy(c):>6.3f}  {np.mean(lm):>8.4f}  {np.mean(tc):>8.2f}s")
print("-"*72)
print(f"\nDivergencia B/H1 vs B/H2        : {div_B1B2:.1f}%  (historia distinta)")
print(f"Divergencia C/H1 vs C/H2        : {div_C1C2_rate:.1f}%  (historia distinta)")
print(f"Divergencia B/H1 vs C/H1        : {div_BC1_rate:.1f}%  (misma historia, mecanismo distinto)")
print(f"Divergencia B/H2 vs C/H2        : {div_BC2_rate:.1f}%  (misma historia, mecanismo distinto)")
print("="*72)
print()
print("Interpretación:")
print(f"  B vs C con MISMA historia     : {(div_BC1_rate+div_BC2_rate)/2:.1f}% divergencia")
print("  → Si > 0%: los mecanismos producen comportamiento cualitativamente distinto")
print("  → Si = 0%: la deformación de V y el buffer son equivalentes operacionalmente")

# ─────────────────────────────────────────────────────────────
# FIGURA
# ─────────────────────────────────────────────────────────────
BG="#0D1B2A"; W="#FFFFFF"; CA="#4FC3F7"; CB="#FF7043"; CC="#A5D6A7"; CG="#FFD54F"

fig=plt.figure(figsize=(16,10),facecolor=BG)
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.38)
ax1=fig.add_subplot(gs[0,:2])   # distribución atractores (ancho)
ax2=fig.add_subplot(gs[0,2])    # entropía
ax3=fig.add_subplot(gs[1,0])    # τ_crítico
ax4=fig.add_subplot(gs[1,1])    # λ_min
ax5=fig.add_subplot(gs[1,2])    # divergencia B vs C misma historia

def sty(ax, title="", xl="", yl=""):
    ax.set_facecolor(BG); ax.tick_params(colors=W,labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor("#334466")
    ax.set_xlabel(xl,color=W,fontsize=9); ax.set_ylabel(yl,color=W,fontsize=9)
    if title: ax.set_title(title,color=W,fontsize=10,fontweight="bold")
    ax.grid(True,color="#1A3A5A",lw=0.5,alpha=0.5)

x=np.arange(4); ww=0.22
ax1.bar(x-ww,   cA,  width=ww, color=CA, alpha=0.9, label="A — gradiente puro")
ax1.bar(x,      cB1, width=ww, color=CB, alpha=0.9, label="B/H₁ — historia estructural (IGU)")
ax1.bar(x+ww,   cC1, width=ww, color=CC, alpha=0.9, label="C/H₁ — memoria explícita (control)")
ax1.set_xticks(x)
ax1.set_xticklabels(["a0 [-2,-2]","a1 [2,-2]","a2 [-2,2]","a3 [2,2]"],color=W,fontsize=9)
ax1.set_ylabel("Frecuencia",color=W,fontsize=9)
ax1.legend(fontsize=8,facecolor="#1A2A3A",labelcolor=W,framealpha=0.9)
for bar in ax1.patches:
    h=bar.get_height()
    if h>2: ax1.text(bar.get_x()+bar.get_width()/2,h+1,str(int(h)),
                     ha='center',va='bottom',color=W,fontsize=7)
sty(ax1, f"Distribución de atractores finales  (N={N})")

bars=ax2.bar(["A","B/H₁","C/H₁"],
             [entropy(cA),entropy(cB1),entropy(cC1)],
             color=[CA,CB,CC],alpha=0.9,width=0.5)
ax2.axhline(np.log(4),color=W,lw=0.8,ls='--',alpha=0.4)
ax2.set_ylim(0,np.log(4)*1.2)
for b,v in zip(bars,[entropy(cA),entropy(cB1),entropy(cC1)]):
    ax2.text(b.get_x()+b.get_width()/2,v+0.01,f"{v:.3f}",
             ha='center',va='bottom',color=W,fontsize=9)
sty(ax2,"Entropía de selección",yl="H (nats)")

ax3.boxplot([tcA,tcB1,tcC1],patch_artist=True,
            medianprops=dict(color=W,lw=2),
            whiskerprops=dict(color=W),capprops=dict(color=W),
            flierprops=dict(markerfacecolor=W,marker='o',markersize=3))
for patch,col in zip(ax3.patches,[CA,CB,CC]): patch.set_facecolor(col); patch.set_alpha(0.8)
ax3.set_xticklabels(["A","B/H₁","C/H₁"],color=W,fontsize=9)
ax3.axhline(0,color=W,lw=0.5,ls=':',alpha=0.3)
sty(ax3,"τ_crítico por simulación (s)",yl="segundos")

ax4.boxplot([lmA,lmB1,lmC1],patch_artist=True,
            medianprops=dict(color=W,lw=2),
            whiskerprops=dict(color=W),capprops=dict(color=W),
            flierprops=dict(markerfacecolor=W,marker='o',markersize=3))
for patch,col in zip(ax4.patches,[CA,CB,CC]): patch.set_facecolor(col); patch.set_alpha(0.8)
ax4.set_xticklabels(["A","B/H₁","C/H₁"],color=W,fontsize=9)
ax4.axhline(0,color='#FFD54F',lw=0.8,ls='--',alpha=0.5)
sty(ax4,"λ_min medio por simulación")

# Panel clave: divergencia B vs C con la MISMA historia
labels_div=["B/H₁ vs C/H₁\n(misma historia)","B/H₂ vs C/H₂\n(misma historia)"]
vals_div=[div_BC1_rate, div_BC2_rate]
bars5=ax5.bar(labels_div,vals_div,color=[CB,CC],alpha=0.9,width=0.5)
ax5.axhline(50,color=W,lw=0.8,ls='--',alpha=0.4)
ax5.set_ylim(0,105)
for b,v in zip(bars5,vals_div):
    ax5.text(b.get_x()+b.get_width()/2,v+1.5,f"{v:.0f}%",
             ha='center',va='bottom',color=W,fontsize=10,fontweight='bold')
ax5.text(0.5,55,"azar puro",color=W,fontsize=8,alpha=0.5,ha='center')
sty(ax5,"B vs C con la MISMA historia\n(divergencia = mecanismos distintos)",yl="%")

fig.suptitle(
    f"IGU — Comparación A (gradiente) vs B (historia estructural) vs C (memoria explícita)  |  N={N}",
    color=W,fontsize=12,fontweight="bold")

plt.savefig("igu_ABC_comparison.png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.show()
print("\nFigura guardada: igu_ABC_comparison.png")

from google.colab import files
files.download("igu_ABC_comparison.png")