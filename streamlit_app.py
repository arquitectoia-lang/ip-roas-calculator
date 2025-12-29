# -*- coding: utf-8 -*-
"""
IP-ROAS Calculator - Versión Web Streamlit
===========================================
Calculadora de ROAS basado en Inversión Publicitaria

Deploy en Streamlit Cloud:
1. Subir a GitHub
2. Conectar en share.streamlit.io
3. Configurar Secrets con credenciales
4. ¡Listo!

Autor: Juan Pablo Fernández Gutiérrez
Área: Tecnología - SaleADS.ai
Versión: 1.2 (basada en GUI v4.1) - Con autenticación y carga CSV mejorada
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import hmac
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Calculadora IP-ROAS | SaleADS.ai",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SISTEMA DE AUTENTICACIÓN
# ============================================================================

def check_password():
    """Verifica credenciales del usuario. Retorna True si está autenticado."""
    
    def password_entered():
        """Valida usuario y contraseña ingresados."""
        try:
            # Obtener credenciales de secrets
            users = st.secrets["credentials"]["usernames"]
            
            username = st.session_state["username"]
            password = st.session_state["password"]
            
            if username in users:
                stored_password = users[username]["password"]
                # Comparar contraseñas (soporta hash SHA256 o texto plano)
                if stored_password.startswith("sha256:"):
                    # Contraseña hasheada
                    hash_stored = stored_password.replace("sha256:", "")
                    hash_input = hashlib.sha256(password.encode()).hexdigest()
                    if hmac.compare_digest(hash_stored, hash_input):
                        st.session_state["authenticated"] = True
                        st.session_state["current_user"] = username
                        st.session_state["user_name"] = users[username].get("name", username)
                        del st.session_state["password"]
                        return
                else:
                    # Contraseña en texto plano (para desarrollo)
                    if password == stored_password:
                        st.session_state["authenticated"] = True
                        st.session_state["current_user"] = username
                        st.session_state["user_name"] = users[username].get("name", username)
                        del st.session_state["password"]
                        return
            
            st.session_state["authenticated"] = False
            st.error("😕 Usuario o contraseña incorrectos")
        except KeyError:
            st.error("⚠️ Error de configuración: Credenciales no encontradas en Secrets")
            st.session_state["authenticated"] = False
    
    # Si ya está autenticado, retornar True
    if st.session_state.get("authenticated", False):
        return True
    
    # Mostrar formulario de login
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(135deg, #1a1a3e 0%, #2a2a5e 100%);
            border-radius: 16px;
            margin-top: 5rem;
        }
        .login-title {
            color: #a78bfa;
            font-size: 1.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        .login-subtitle {
            color: #94a3b8;
            text-align: center;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-top: 3rem;">
            <h1 style="color: #a78bfa;">📊 Calculadora IP-ROAS</h1>
            <p style="color: #94a3b8;">SaleADS.ai — Acceso restringido</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.text_input("👤 Usuario", key="username")
            st.text_input("🔒 Contraseña", type="password", key="password")
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Ingresar", use_container_width=True)
            if submitted:
                password_entered()
        
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; color: #64748b; font-size: 0.8rem;">
            Contacte al administrador si necesita acceso
        </div>
        """, unsafe_allow_html=True)
    
    return False


def logout():
    """Cierra la sesión del usuario."""
    st.session_state["authenticated"] = False
    st.session_state["current_user"] = None
    st.session_state["user_name"] = None
    st.rerun()


# ============================================================================
# VERIFICAR AUTENTICACIÓN ANTES DE MOSTRAR LA APP
# ============================================================================

if not check_password():
    st.stop()

# ============================================================================
# ESTILOS CSS PERSONALIZADOS
# ============================================================================

st.markdown("""
<style>
    /* Ocultar menú hamburguesa y footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header principal */
    .main-header {
        background: linear-gradient(90deg, #1a1a3e 0%, #2a2a5e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .main-title {
        color: #a78bfa;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    
    .main-subtitle {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    /* Tarjetas de métricas */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid;
    }
    
    .metric-card.purple { border-left-color: #a78bfa; }
    .metric-card.green { border-left-color: #10b981; }
    .metric-card.orange { border-left-color: #f97316; }
    .metric-card.cyan { border-left-color: #06b6d4; }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: white;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    /* Producto crítico */
    .critical-product {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid #f97316;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .critical-title {
        color: #f97316;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MOTOR DE CÁLCULO IP-ROAS
# ============================================================================

@dataclass
class Producto:
    """Representa un producto del portafolio del cliente."""
    nombre: str
    precio: float
    margen_bruto: float  # Porcentaje como decimal (0.30 = 30%)
    
    @property
    def margen_absoluto(self) -> float:
        return self.precio * self.margen_bruto


@dataclass
class ParametrosCliente:
    """Parámetros de entrada para el cálculo IP-ROAS."""
    inversion_publicitaria: float  # IP
    tarifa_fija: float             # TF
    ingreso_esperado: float        # IE
    productos: List[Producto] = field(default_factory=list)
    
    @property
    def costos_totales(self) -> float:
        return self.inversion_publicitaria + self.tarifa_fija + self.ingreso_esperado
    
    @property
    def producto_margen_minimo(self) -> Optional[Producto]:
        if not self.productos:
            return None
        return min(self.productos, key=lambda p: p.margen_absoluto)


@dataclass
class ResultadosIPROAS:
    """Resultados del cálculo IP-ROAS."""
    ip_roas: float
    vum: int
    roas_min_tradicional: float
    cpr_estimado: float
    costos_totales: float
    margen_minimo_usado: float
    precio_producto_minimo: float
    producto_critico: str


class CalculadoraIPROAS:
    """Motor de cálculo para la metodología IP-ROAS."""
    
    def __init__(self, parametros: ParametrosCliente):
        self.parametros = parametros
    
    def calcular_ip_roas(self) -> float:
        IP = self.parametros.inversion_publicitaria
        TF = self.parametros.tarifa_fija
        IE = self.parametros.ingreso_esperado
        if IP <= 0:
            return float('inf')
        return 1 + (TF + IE) / IP
    
    def calcular_vum(self, margen_absoluto: Optional[float] = None) -> int:
        if margen_absoluto is None:
            producto = self.parametros.producto_margen_minimo
            if producto is None or producto.margen_absoluto <= 0:
                return 0
            margen_absoluto = producto.margen_absoluto
        if margen_absoluto <= 0:
            return 0
        return math.ceil(self.parametros.costos_totales / margen_absoluto)
    
    def calcular_roas_minimo_tradicional(self, vum: Optional[int] = None, precio: Optional[float] = None) -> float:
        if vum is None:
            vum = self.calcular_vum()
        if precio is None:
            producto = self.parametros.producto_margen_minimo
            if producto is None:
                return 0.0
            precio = producto.precio
        IP = self.parametros.inversion_publicitaria
        if IP <= 0:
            return float('inf')
        return (precio * vum) / IP
    
    def calcular_cpr_estimado(self, vum: Optional[int] = None) -> float:
        if vum is None:
            vum = self.calcular_vum()
        if vum <= 0:
            return float('inf')
        return self.parametros.inversion_publicitaria / vum
    
    def calcular_todo(self) -> ResultadosIPROAS:
        producto = self.parametros.producto_margen_minimo
        if producto is None:
            return ResultadosIPROAS(0, 0, 0, 0, 0, 0, 0, "N/A")
        
        ip_roas = self.calcular_ip_roas()
        vum = self.calcular_vum()
        roas_trad = self.calcular_roas_minimo_tradicional(vum)
        cpr = self.calcular_cpr_estimado(vum)
        
        return ResultadosIPROAS(
            ip_roas=ip_roas, vum=vum, roas_min_tradicional=roas_trad, cpr_estimado=cpr,
            costos_totales=self.parametros.costos_totales, margen_minimo_usado=producto.margen_absoluto,
            precio_producto_minimo=producto.precio, producto_critico=producto.nombre
        )
    
    @staticmethod
    def f_ip_roas(IP: float, TF: float, IE: float) -> float:
        if IP <= 0:
            return float('inf')
        return 1 + (TF + IE) / IP
    
    @staticmethod
    def f_roas_tradicional(IP: float, TF: float, IE: float, m_star: float, p_star: float) -> float:
        if IP <= 0 or m_star <= 0:
            return float('inf')
        vum = math.ceil((TF + IP + IE) / m_star)
        return (p_star * vum) / IP


# ============================================================================
# INICIALIZACIÓN DEL ESTADO
# ============================================================================

if 'productos' not in st.session_state:
    st.session_state.productos = []

if 'ip' not in st.session_state:
    st.session_state.ip = 0.0

if 'tf' not in st.session_state:
    st.session_state.tf = 0.0

if 'ie' not in st.session_state:
    st.session_state.ie = 0.0


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def crear_grafico_sensibilidad(x_values, ip_roas_values, roas_trad_values, titulo, xlabel):
    """Crea gráfico de sensibilidad con Plotly."""
    fig = go.Figure()
    
    # Línea IP-ROAS
    fig.add_trace(go.Scatter(
        x=x_values,
        y=ip_roas_values,
        mode='lines',
        name='IP-ROAS',
        line=dict(color='#a78bfa', width=2),
        hovertemplate=f'{xlabel}: %{{x:,.2f}}<br>IP-ROAS: %{{y:.4f}}<extra></extra>'
    ))
    
    # Línea ROAS Tradicional
    fig.add_trace(go.Scatter(
        x=x_values,
        y=roas_trad_values,
        mode='lines',
        name='ROAS Tradicional',
        line=dict(color='#f97316', width=2),
        hovertemplate=f'{xlabel}: %{{x:,.2f}}<br>ROAS Trad: %{{y:.4f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=titulo, font=dict(color='white', size=16)),
        xaxis=dict(
            title=xlabel,
            gridcolor='#333',
            tickformat=',.0f' if 'Margen' not in xlabel else '.1%'
        ),
        yaxis=dict(title='ROAS', gridcolor='#333'),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#94a3b8'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    
    return fig


def calcular_sensibilidad(tipo: str, parametros: ParametrosCliente, num_points: int = 50):
    """Calcula datos de sensibilidad para un parámetro."""
    producto = parametros.producto_margen_minimo
    if producto is None:
        return None, None, None
    
    m_star = producto.margen_absoluto
    p_star = producto.precio
    IP = parametros.inversion_publicitaria
    TF = parametros.tarifa_fija
    IE = parametros.ingreso_esperado
    
    if tipo == "IP":
        if IP <= 0:
            IP = 10000
        x_min, x_max = max(100, IP * 0.5), IP * 1.5
        x_values = np.linspace(x_min, x_max, num_points)
        ip_roas_values = [CalculadoraIPROAS.f_ip_roas(x, TF, IE) for x in x_values]
        roas_trad_values = [CalculadoraIPROAS.f_roas_tradicional(x, TF, IE, m_star, p_star) for x in x_values]
        
    elif tipo == "TF":
        if TF <= 0:
            TF = 5000
        x_min, x_max = max(0, TF * 0.5), TF * 1.5
        x_values = np.linspace(x_min, x_max, num_points)
        ip_roas_values = [CalculadoraIPROAS.f_ip_roas(IP, x, IE) for x in x_values]
        roas_trad_values = [CalculadoraIPROAS.f_roas_tradicional(IP, x, IE, m_star, p_star) for x in x_values]
        
    elif tipo == "IE":
        if IE <= 0:
            IE = 5000
        x_min, x_max = max(0, IE * 0.5), IE * 1.5
        x_values = np.linspace(x_min, x_max, num_points)
        ip_roas_values = [CalculadoraIPROAS.f_ip_roas(IP, TF, x) for x in x_values]
        roas_trad_values = [CalculadoraIPROAS.f_roas_tradicional(IP, TF, x, m_star, p_star) for x in x_values]
        
    elif tipo == "Margen":
        margen_pct = producto.margen_bruto
        if margen_pct <= 0:
            margen_pct = 0.3
        x_min, x_max = max(0.05, margen_pct * 0.5), min(0.95, margen_pct * 1.5)
        x_values = np.linspace(x_min, x_max, num_points)
        ip_roas_values = [CalculadoraIPROAS.f_ip_roas(IP, TF, IE) for _ in x_values]  # IP-ROAS no depende del margen
        roas_trad_values = [CalculadoraIPROAS.f_roas_tradicional(IP, TF, IE, p_star * x, p_star) for x in x_values]
    
    else:
        return None, None, None
    
    return x_values, ip_roas_values, roas_trad_values


# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <p class="main-title">📊 Calculadora IP-ROAS</p>
    <p class="main-subtitle">SaleADS.ai — Metodología IP-ROAS v1.2</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Parámetros de entrada
with st.sidebar:
    # Mostrar usuario y botón logout
    st.markdown(f"""
    <div style="background: #1e1e2e; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">
        <span style="color: #10b981;">👤 {st.session_state.get('user_name', 'Usuario')}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Cerrar sesión", use_container_width=True):
        logout()
    
    st.divider()
    
    st.header("⚙️ Parámetros de Entrada")
    
    st.markdown("##### 💰 Inversión Publicitaria (IP)")
    ip = st.number_input(
        "Presupuesto negociado con el cliente",
        min_value=0.0,
        value=st.session_state.ip,
        step=1000.0,
        format="%.2f",
        key="ip_input",
        label_visibility="collapsed"
    )
    st.session_state.ip = ip
    
    st.markdown("##### 🏢 Tarifa Fija (TF)")
    tf = st.number_input(
        "Fee de agencia o costos fijos",
        min_value=0.0,
        value=st.session_state.tf,
        step=500.0,
        format="%.2f",
        key="tf_input",
        label_visibility="collapsed"
    )
    st.session_state.tf = tf
    
    st.markdown("##### 📈 Ingreso Esperado (IE)")
    ie = st.number_input(
        "Utilidad objetivo de la campaña",
        min_value=0.0,
        value=st.session_state.ie,
        step=500.0,
        format="%.2f",
        key="ie_input",
        label_visibility="collapsed"
    )
    st.session_state.ie = ie
    
    st.divider()
    
    # Cargar CSV
    st.header("📁 Cargar Productos (CSV)")
    uploaded_file = st.file_uploader(
        "Formato: nombre,precio,margen",
        type=['csv'],
        help="CSV con columnas: nombre, precio, margen. Puede incluir encabezado o no."
    )
    
    if uploaded_file is not None:
        try:
            # Leer el archivo como texto primero para analizar
            import io
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.strip().split('\n')
            
            productos_nuevos = []
            
            for i, line in enumerate(lines):
                # Separar por coma
                parts = [p.strip().strip('"').strip("'") for p in line.split(',')]
                
                if len(parts) >= 3:
                    nombre_raw = parts[0]
                    precio_raw = parts[1]
                    margen_raw = parts[2]
                    
                    # Detectar si es fila de encabezado (primera fila con texto no numérico en precio)
                    try:
                        precio = float(precio_raw.replace('$', '').replace(',', ''))
                        margen = float(margen_raw.replace('%', '').replace(',', ''))
                        
                        # Si margen > 1, asumir que es porcentaje (ej: 35 -> 0.35)
                        if margen > 1:
                            margen = margen / 100
                        
                        productos_nuevos.append(Producto(nombre_raw, precio, margen))
                    except ValueError:
                        # Es una fila de encabezado, saltar
                        continue
            
            if productos_nuevos:
                st.session_state.productos = productos_nuevos
                st.success(f"✅ {len(productos_nuevos)} producto(s) cargado(s)")
            else:
                st.warning("⚠️ No se encontraron productos válidos en el CSV")
                
        except Exception as e:
            st.error(f"Error al cargar CSV: {e}")
    
    st.divider()
    
    # Agregar producto manual
    st.header("📦 Agregar Producto")
    with st.form("add_product"):
        prod_nombre = st.text_input("Nombre del producto")
        prod_precio = st.number_input("Precio ($)", min_value=0.0, step=100.0)
        prod_margen = st.number_input("Margen bruto (%)", min_value=0.0, max_value=100.0, step=5.0)
        
        if st.form_submit_button("➕ Agregar"):
            if prod_nombre and prod_precio > 0 and prod_margen > 0:
                nuevo_producto = Producto(prod_nombre, prod_precio, prod_margen / 100)
                st.session_state.productos.append(nuevo_producto)
                st.success(f"✅ '{prod_nombre}' agregado")
                st.rerun()


# Calcular resultados
parametros = ParametrosCliente(
    inversion_publicitaria=st.session_state.ip,
    tarifa_fija=st.session_state.tf,
    ingreso_esperado=st.session_state.ie,
    productos=st.session_state.productos
)

calculadora = CalculadoraIPROAS(parametros)
resultados = calculadora.calcular_todo()

# Tarjetas de métricas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card purple">
        <div class="metric-value">{resultados.ip_roas:.4f}</div>
        <div class="metric-label">IP-ROAS<br>¿Cuánto debe generar cada peso invertido?</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card green">
        <div class="metric-value">{resultados.vum:,} uds</div>
        <div class="metric-label">VUM<br>¿Cuántas unidades debo vender?</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card orange">
        <div class="metric-value">{resultados.roas_min_tradicional:.4f}</div>
        <div class="metric-label">ROAS Tradicional<br>¿Retorno mínimo en ventas totales?</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card cyan">
        <div class="metric-value">${resultados.cpr_estimado:,.2f}</div>
        <div class="metric-label">CPR Estimado<br>¿Cuánto me cuesta cada venta?</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Contenido principal
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Resumen", 
    "📈 Sens. IP", 
    "📈 Sens. TF", 
    "📈 Sens. IE",
    "📈 Sens. Margen"
])

with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📦 Portafolio de Productos")
        
        if st.session_state.productos:
            df_productos = pd.DataFrame([
                {
                    'Nombre': p.nombre,
                    'Precio': f"${p.precio:,.2f}",
                    'Margen %': f"{p.margen_bruto*100:.1f}%",
                    'Margen $': f"${p.margen_absoluto:,.2f}"
                }
                for p in st.session_state.productos
            ])
            st.dataframe(df_productos, use_container_width=True, hide_index=True)
            
            if st.button("🗑️ Limpiar productos"):
                st.session_state.productos = []
                st.rerun()
        else:
            st.info("No hay productos cargados. Use el sidebar para agregar productos.")
    
    with col_right:
        st.subheader("⚠️ Producto Crítico")
        
        producto_critico = parametros.producto_margen_minimo
        if producto_critico:
            st.markdown(f"""
            <div class="critical-product">
                <div class="critical-title">⚠️ {producto_critico.nombre}</div>
                <p>📛 <strong>Nombre:</strong> {producto_critico.nombre}</p>
                <p>💵 <strong>Precio:</strong> ${producto_critico.precio:,.2f}</p>
                <p>📊 <strong>Margen %:</strong> {producto_critico.margen_bruto*100:.1f}%</p>
                <p>💰 <strong>Margen $:</strong> ${producto_critico.margen_absoluto:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 Este es el producto con menor margen absoluto. Los cálculos de VUM, ROAS Tradicional y CPR se basan en este producto.")
        else:
            st.warning("Agregue al menos un producto para ver el producto crítico.")
        
        st.subheader("📊 Resumen de Costos")
        st.metric("Costos Totales (IP + TF + IE)", f"${parametros.costos_totales:,.2f}")

with tab2:
    st.subheader("Sensibilidad a la Inversión Publicitaria (IP)")
    if st.session_state.productos:
        x_vals, ip_roas_vals, roas_trad_vals = calcular_sensibilidad("IP", parametros)
        if x_vals is not None:
            fig = crear_grafico_sensibilidad(
                x_vals, ip_roas_vals, roas_trad_vals,
                "Sensibilidad IP-ROAS vs Inversión Publicitaria",
                "Inversión Publicitaria ($)"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Agregue productos para ver el análisis de sensibilidad.")

with tab3:
    st.subheader("Sensibilidad a la Tarifa Fija (TF)")
    if st.session_state.productos:
        x_vals, ip_roas_vals, roas_trad_vals = calcular_sensibilidad("TF", parametros)
        if x_vals is not None:
            fig = crear_grafico_sensibilidad(
                x_vals, ip_roas_vals, roas_trad_vals,
                "Sensibilidad IP-ROAS vs Tarifa Fija",
                "Tarifa Fija ($)"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Agregue productos para ver el análisis de sensibilidad.")

with tab4:
    st.subheader("Sensibilidad al Ingreso Esperado (IE)")
    if st.session_state.productos:
        x_vals, ip_roas_vals, roas_trad_vals = calcular_sensibilidad("IE", parametros)
        if x_vals is not None:
            fig = crear_grafico_sensibilidad(
                x_vals, ip_roas_vals, roas_trad_vals,
                "Sensibilidad IP-ROAS vs Ingreso Esperado",
                "Ingreso Esperado ($)"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Agregue productos para ver el análisis de sensibilidad.")

with tab5:
    st.subheader("Sensibilidad al Margen Bruto")
    if st.session_state.productos:
        x_vals, ip_roas_vals, roas_trad_vals = calcular_sensibilidad("Margen", parametros)
        if x_vals is not None:
            fig = crear_grafico_sensibilidad(
                x_vals, ip_roas_vals, roas_trad_vals,
                "Sensibilidad IP-ROAS vs Margen Bruto",
                "Margen Bruto (%)"
            )
            # Ajustar formato del eje X para porcentajes
            fig.update_xaxes(tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Agregue productos para ver el análisis de sensibilidad.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem;">
    📊 Calculadora IP-ROAS v1.2 | SaleADS.ai — Metodología IP-ROAS<br>
    Desarrollado por Juan Pablo Fernández Gutiérrez | Área de Tecnología
</div>
""", unsafe_allow_html=True)
