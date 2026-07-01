"""Interface Python ↔ XFoil via subprocess.
"""

from __future__ import annotations

import os
import re
import time
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List


if os.name == "nt":
    _CREATIONFLAGS = subprocess.CREATE_NO_WINDOW   # type: ignore[attr-defined]
else:
    _CREATIONFLAGS = 0


class XFoilRunner:
    def __init__(
        self,
        xfoil_path: str,
        work_dir: Path,
        timeout: float = 30.0,
        n_iter: int = 200,
        ncrit: float = 9.0,
        xtr_top: float = 1.0,
        xtr_bot: float = 1.0,
        debug: bool = False,
        visible: bool = False,
        visible_delay: float = 0.0,
    ) -> None:
        self.xfoil_path = str(xfoil_path)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.n_iter = n_iter
        self.ncrit = ncrit
        self.xtr_top = xtr_top
        self.xtr_bot = xtr_bot
        self.debug = debug
        self.visible = visible
        self.visible_delay = visible_delay

        self._last_stdout: str = ""
        self._last_stderr: str = ""

        if not Path(self.xfoil_path).exists():
            print(f"[XFoilRunner] AVISO: binário não encontrado em '{self.xfoil_path}'")

    def run_alpha(
        self,
        dat_file: Path,
        alpha: float,
        reynolds: float,
        mach: float = 0.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Roda em ângulo fixo. Retorna (Cl, Cd) ou (None, None) se falhar."""
        dat_file = Path(dat_file)
        cwd = dat_file.parent
        polar = self._unique_polar_path(cwd)

        # NOMES CURTOS: passa só o nome do arquivo, não o caminho completo
        cmds = self._build_cmd_alpha(dat_file.name, polar.name, alpha, reynolds, mach)
        ok = self._exec_xfoil(cmds, cwd=cwd)
        if not ok:
            self._maybe_print_debug(cmds)
            self._cleanup(polar)
            return None, None
        results = self._parse_polar(polar)
        self._cleanup(polar)
        if not results:
            self._maybe_print_debug(cmds)
            return None, None
        _, cl, cd = results[-1]
        if cd <= 0 or not (0 < abs(cl) < 5):
            return None, None
        return cl, cd

    def run_alpha_sweep(
        self,
        dat_file: Path,
        alpha_start: float,
        alpha_end: float,
        alpha_step: float,
        reynolds: float,
        mach: float = 0.0,
    ) -> List[Tuple[float, float, float]]:
        """ASEQ — XFoil reaproveita a solução do α anterior como chute inicial."""
        dat_file = Path(dat_file)
        cwd = dat_file.parent
        polar = self._unique_polar_path(cwd)

        cmds = self._build_cmd_sweep(
            dat_file.name, polar.name,
            alpha_start, alpha_end, alpha_step, reynolds, mach,
        )
        ok = self._exec_xfoil(cmds, cwd=cwd)
        if not ok:
            self._maybe_print_debug(cmds)
            self._cleanup(polar)
            return []
        results = self._parse_polar(polar)
        self._cleanup(polar)
        if not results:
            self._maybe_print_debug(cmds)
        return results

    def diagnose(
        self,
        dat_file: Path,
        alpha: float = 5.0,
        reynolds: float = 500_000.0,
        mach: float = 0.0,
    ) -> bool:
        """Modo verboso pra validar instalação/protocolo. Use quando o baseline falhar."""
        dat_file = Path(dat_file)
        cwd = dat_file.parent
        polar = self._unique_polar_path(cwd)
        cmds = self._build_cmd_alpha(dat_file.name, polar.name, alpha, reynolds, mach)

        print("\n" + "-" * 76)
        print("  XFOIL DIAGNOSE")
        print("-" * 76)
        print(f"  Executavel:    {self.xfoil_path}")
        print(f"  Existe?        {Path(self.xfoil_path).exists()}")
        print(f"  Working dir:   {cwd}")
        print(f"  Arquivo .dat:  {dat_file.name}  (existe? {dat_file.exists()})")
        print(f"  alpha = {alpha} grau  |  Re = {reynolds:.0f}  |  M = {mach}")
        print("\n  -> Comandos enviados ao XFoil:")
        for ln in cmds.rstrip("\n").split("\n"):
            tag = "(EOL)" if ln == "" else ""
            print(f"      | {ln} {tag}")

        try:
            proc = subprocess.run(
                [self.xfoil_path],
                input=cmds, text=True, capture_output=True,
                timeout=self.timeout, cwd=str(cwd),
                creationflags=_CREATIONFLAGS,
            )
            self._last_stdout = proc.stdout or ""
            self._last_stderr = proc.stderr or ""
            print(f"\n  <- returncode = {proc.returncode}")
            tail = self._last_stdout.splitlines()[-30:]
            print("  <- stdout do XFoil (ultimas 30 linhas):")
            for ln in tail:
                print(f"      | {ln}")
            if self._last_stderr.strip():
                print("  <- stderr:")
                for ln in self._last_stderr.splitlines():
                    print(f"      | {ln}")
        except subprocess.TimeoutExpired:
            print(f"\n  [X] TIMEOUT apos {self.timeout}s")
            return False
        except FileNotFoundError as e:
            print(f"\n  [X] EXECUTAVEL NAO ENCONTRADO: {e}")
            return False
        except Exception as e:
            print(f"\n  [X] ERRO {type(e).__name__}: {e}")
            return False

        if polar.exists():
            print(f"\n  [OK] Polar gerado em '{polar.name}' ({polar.stat().st_size} bytes):")
            print(polar.read_text())
            results = self._parse_polar(polar)
            try: polar.unlink()
            except Exception: pass
            if results:
                _, cl, cd = results[-1]
                print(f"  [OK] Resultado: Cl = {cl:.4f}  |  Cd = {cd:.5f}  |  Cl/Cd = {cl/cd:.2f}")
            print("-" * 76)
            return bool(results)
        else:
            print("\n  [X] Polar NAO foi gerado.")
            print("    Causas comuns:")
            print("      1) Caminho do .dat ainda muito longo (cwd precisa ser curto).")
            print("      2) Arquivo .dat malformado (1a linha precisa ser nome do perfil).")
            print("      3) XFoil nao converge para a geometria/Re fornecidos.")
            print("-" * 76)
            return False

    def _build_cmd_alpha(
        self,
        dat_name: str,
        polar_name: str,
        alpha: float,
        re: float,
        mach: float,
    ) -> str:
        # PANE depois do LOAD é crítico: o XFoil repaneliza com seu próprio
        # algoritmo. Sem isso, perfis CST com amostragem cosenoidal pura
        # produzem painéis com curvatura inconsistente -> Cp oscila -> BL
        # não converge. Foi a causa das falhas em massa.
        cmds: List[str] = [
            f"LOAD {dat_name}",
            "PANE",
            "OPER",
            "VPAR",
            f"N {self.ncrit:.1f}",
            "",
        ]
        if re > 0:
            cmds.append(f"VISC {re:.0f}")
        if mach > 0:
            cmds.append(f"MACH {mach:.3f}")
        cmds.extend([
            "PACC",
            polar_name,
            "",
            f"ALFA {alpha:.3f}",
            "",
            "QUIT",
        ])
        return self._wrap(cmds)

    def _build_cmd_sweep(
        self,
        dat_name: str, polar_name: str,
        a0: float, a1: float, da: float,
        re: float, mach: float,
    ) -> str:
        # ASEQ direto, uma passada só. Com PANE o BL converge em 5-15 iters
        # por alpha; sem PANE o sweep falhava.
        cmds: List[str] = [
            f"LOAD {dat_name}",
            "PANE",
            "OPER",
            "VPAR",
            f"N {self.ncrit:.1f}",
            "",
        ]
        if re > 0:
            cmds.append(f"VISC {re:.0f}")
        if mach > 0:
            cmds.append(f"MACH {mach:.3f}")
        cmds.extend([
            "PACC",
            polar_name,
            "",
            f"ASEQ {a0:.3f} {a1:.3f} {abs(da):.3f}",
            "",
            "QUIT",
        ])
        return self._wrap(cmds)

    def _wrap(self, commands: List[str]) -> str:
        # PLOP G = headless. Em modo visível mantém os gráficos ligados.
        if self.visible:
            header: List[str] = []
        else:
            header = ["PLOP", "G", ""]
        return "\n".join(header + commands) + "\n"

    def _exec_xfoil(self, cmd_str: str, cwd: Path) -> bool:
        """Executa o XFoil. NUNCA propaga exceção."""
        if os.name == "nt":
            flags = 0 if self.visible else _CREATIONFLAGS
        else:
            flags = 0
        try:
            proc = subprocess.run(
                [self.xfoil_path],
                input=cmd_str, text=True, capture_output=True,
                timeout=self.timeout, cwd=str(cwd),
                creationflags=flags,
            )
            self._last_stdout = proc.stdout or ""
            self._last_stderr = proc.stderr or ""
            if self.visible_delay > 0:
                time.sleep(self.visible_delay)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            self._last_stdout = "(TIMEOUT)"
            return False
        except FileNotFoundError:
            self._last_stdout = "(EXECUTABLE NOT FOUND)"
            return False
        except Exception as e:                         # noqa: BLE001
            self._last_stdout = f"(EXCEPTION: {type(e).__name__})"
            return False

    def _maybe_print_debug(self, cmds: str) -> None:
        if not self.debug:
            return
        print("\n[XFoil DEBUG] ultima execucao falhou. Comandos enviados:")
        for ln in cmds.rstrip("\n").split("\n"):
            print(f"   | {ln}")
        print("[XFoil DEBUG] ultimas 15 linhas do stdout:")
        for ln in (self._last_stdout or "").splitlines()[-15:]:
            print(f"   | {ln}")

    @staticmethod
    def _parse_polar(polar_file: Path) -> List[Tuple[float, float, float]]:
        if not polar_file.exists():
            return []
        try:
            with open(polar_file, "r") as f:
                lines = f.readlines()
        except Exception:
            return []

        # Detecta início dos dados (linha de tracejados)
        data_start = None
        for i, ln in enumerate(lines):
            if re.match(r"^\s*-+\s+-+", ln):
                data_start = i + 1
                break
        if data_start is None:
            return []

        results: List[Tuple[float, float, float]] = []
        for ln in lines[data_start:]:
            parts = ln.split()
            if len(parts) >= 3:
                try:
                    a = float(parts[0])
                    cl = float(parts[1])
                    cd = float(parts[2])
                    if all(map(lambda v: v == v, [a, cl, cd])):    # not NaN
                        results.append((a, cl, cd))
                except ValueError:
                    continue
        return results

    @staticmethod
    def _unique_polar_path(cwd: Path) -> Path:
        # Nome curto (~14 chars) pra não estourar buffer do XFoil
        tag = f"{(time.time_ns() % 1_000_000):06d}_{os.getpid() % 1000:03d}"
        return Path(cwd) / f"p_{tag}.txt"

    @staticmethod
    def _cleanup(*paths: Path) -> None:
        for p in paths:
            try:
                if p and Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass
