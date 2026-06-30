#!/usr/bin/env python3
"""
Tiago Pro geometric calibration using Figaroh.

Reads a CSV produced by collect_calibration_data.py, runs Figaroh's
Levenberg-Marquardt calibration, and writes the identified joint offsets
to calibration_offset.urdf.xacro.

Usage:
    python3 run_calibration.py --urdf tiago_pro_local.urdf
    python3 run_calibration.py --urdf tiago_pro_local.urdf \\
        --data data/calibration_samples.csv \\
        --output calibration_offset.urdf.xacro
"""

import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

from figaroh.calibration.calibration_tools import (
    get_param_from_yaml,
    add_pee_name,
    load_data,
    calculate_base_kinematics_regressor,
    calc_updated_fkm,
    initialize_variables,
)
from scipy.optimize import least_squares

_HERE         = Path(__file__).parent
_CONFIG       = _HERE / "tiago_pro_calibration_config.yaml"
_DATA_DEFAULT = _HERE / "data" / "calibration_samples.csv"
_OUT_DEFAULT  = _HERE / "calibration_offset.urdf.xacro"
_ROBOT_DESC   = _HERE / "robot_description"


def _pkg_dirs():
    dirs = [str(_ROBOT_DESC)]
    for p in _ROBOT_DESC.iterdir():
        if p.is_dir():
            dirs.append(str(p))
            for sub in p.iterdir():
                if sub.is_dir():
                    dirs.append(str(sub))
    return dirs


class _Robot:
    def __init__(self, m):
        self.model = m
        self.data  = m.createData()
        self.q0    = pin.neutral(m)


# ── Calibration ───────────────────────────────────────────────────────────────

class TiagoProCalibration:

    def __init__(self, robot: _Robot, config_path: str, data_path: str):
        self._robot = robot
        self.model  = robot.model
        self.data   = robot.data

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.param = get_param_from_yaml(robot, config["calibration"])
        self.param["known_baseframe"] = False  # co-estimate mocap->base_footprint transform
        self.param["known_tipframe"]  = False  # estimate marker pos rel. to gripper
        self.param["data_file"]       = data_path

        add_pee_name(self.param)

        self._data_path      = data_path
        self.STATUS          = "NOT CALIBRATED"
        self.calibrated_param: dict = {}

    def load_and_check_data(self) -> None:
        calculate_base_kinematics_regressor([], self.model, self.data, self.param)
        self._pad_csv_missing_joints()
        self.PEE_measured, self.q_measured = load_data(
            self._data_path, self.model, self.param, del_list=[]
        )

    def _pad_csv_missing_joints(self) -> None:
        import pandas as pd, tempfile, os
        joint_headers = [self.model.names[i] for i in self.param["actJoint_idx"]]
        df = pd.read_csv(self._data_path)
        added = [j for j in joint_headers if j not in df.columns]
        if not added:
            return
        for j in added:
            print(f"[info] Column '{j}' missing from CSV — padding with 0.0")
            df[j] = 0.0
        tmp = self._data_path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, self._data_path)
        print(f"\nLoaded {self.param['NbSample']} samples.")
        print(f"Parameters to identify: {self.param['param_name']}\n")

    def solve(self, outlier_eps: float = 0.05, max_iter: int = 10) -> None:
        var0, _ = initialize_variables(self.param, mode=0)
        ci = self.param["calibration_index"]
        tip = self.param.get("tip_pose", None)
        if tip is not None:
            tip_arr = np.atleast_1d(np.array(tip, dtype=float))
            if len(tip_arr) >= ci:
                var0[-ci:] = tip_arr[:ci]

        del_list: list = []
        for iteration in range(max_iter):
            print(f"{'=' * 50}")
            print(f"LM iteration {iteration}")

            result = least_squares(self._cost, var0, method="lm", verbose=1)
            var0   = result.x

            PEEe      = calc_updated_fkm(self.model, self.data, result.x, self.q_measured, self.param)
            residuals = (PEEe - self.PEE_measured).reshape(
                self.param["NbMarkers"] * ci, self.param["NbSample"]
            )
            dist = np.linalg.norm(
                residuals.reshape(-1, 3, self.param["NbSample"]), axis=1
            )
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            mae  = float(np.mean(np.abs(residuals)))
            print(f"RMSE = {rmse*1000:.2f} mm   MAE = {mae*1000:.2f} mm")

            new_outliers = [
                (i, k)
                for i in range(self.param["NbMarkers"])
                for k in range(self.param["NbSample"])
                if dist[i, k] > outlier_eps
            ]
            if new_outliers:
                print(f"Removing {len(new_outliers)} outliers (>{outlier_eps*1000:.0f} mm)")
                del_list += new_outliers
                self.PEE_measured, self.q_measured = load_data(
                    self._data_path, self.model, self.param, del_list=del_list
                )
                var0 = result.x + np.random.normal(0, 0.005, size=result.x.shape)
            else:
                break

        self.calibrated_param = dict(zip(self.param["param_name"], result.x.tolist()))
        self.rmse   = rmse
        self.mae    = mae
        self.STATUS = "CALIBRATED"

        print(f"\n{'=' * 50}")
        print("Calibration results:")
        for name, val in self.calibrated_param.items():
            if "pEE" in name:
                print(f"  {name:40s}  {val*1000:+8.3f} mm")
            else:
                print(f"  {name:40s}  {val*1000:+8.3f} mrad  ({np.degrees(val):+.4f}°)")
        print(f"\nFinal RMSE: {rmse*1000:.2f} mm   MAE: {mae*1000:.2f} mm")

    def _cost(self, var: np.ndarray) -> np.ndarray:
        coeff = self.param.get("coeff_regularize") or 0.01
        PEEe  = calc_updated_fkm(self.model, self.data, var, self.q_measured, self.param)
        ci    = self.param["calibration_index"]
        return np.append(
            self.PEE_measured - PEEe,
            np.sqrt(coeff) * var[:-self.param["NbMarkers"] * ci],
        )


# ── Output ────────────────────────────────────────────────────────────────────

def write_calibration_xacro(calib: TiagoProCalibration, output_path: str) -> None:
    assert calib.STATUS == "CALIBRATED"

    offsets: dict[str, float] = {}
    for param_name, value in calib.calibrated_param.items():
        for jname in [
            "arm_right_1_joint", "arm_right_2_joint", "arm_right_3_joint",
            "arm_right_4_joint", "arm_right_5_joint", "arm_right_6_joint",
            "arm_right_7_joint",
        ]:
            if jname in param_name:
                offsets[f"{jname}_offset"] = value
        if "torso_lift_joint" in param_name:
            offsets["torso_lift_joint_offset"] = value

    xacro_template = """\
<?xml version="1.0"?>
<!--
  Generated by figaroh_tiagoPro/run_calibration.py
  RMSE: {rmse:.2f} mm   MAE: {mae:.2f} mm
-->
<robot xmlns:xacro="http://ros.org/wiki/xacro">

\t<!--Constant parameters-->
\t<xacro:property name="arm_left_1_joint_offset" value="0.0" />
\t<xacro:property name="arm_left_2_joint_offset" value="0.0" />
\t<xacro:property name="arm_left_3_joint_offset" value="0.0" />
\t<xacro:property name="arm_left_4_joint_offset" value="0.0" />
\t<xacro:property name="arm_left_5_joint_offset" value="0.0" />
\t<xacro:property name="arm_left_6_joint_offset" value="0.0" />
\t<xacro:property name="arm_left_7_joint_offset" value="0.0" />

\t<xacro:property name="arm_right_1_joint_offset" value="{arm_right_1_joint_offset}" />
\t<xacro:property name="arm_right_2_joint_offset" value="{arm_right_2_joint_offset}" />
\t<xacro:property name="arm_right_3_joint_offset" value="{arm_right_3_joint_offset}" />
\t<xacro:property name="arm_right_4_joint_offset" value="{arm_right_4_joint_offset}" />
\t<xacro:property name="arm_right_5_joint_offset" value="{arm_right_5_joint_offset}" />
\t<xacro:property name="arm_right_6_joint_offset" value="{arm_right_6_joint_offset}" />
\t<xacro:property name="arm_right_7_joint_offset" value="{arm_right_7_joint_offset}" />

\t<xacro:property name="head_1_joint_offset" value="0.0" />
\t<xacro:property name="head_2_joint_offset" value="0.0" />

</robot>
"""
    values = {
        "rmse": calib.rmse * 1000,
        "mae":  calib.mae  * 1000,
        **{k: 0.0 for k in [
            "arm_right_1_joint_offset", "arm_right_2_joint_offset",
            "arm_right_3_joint_offset", "arm_right_4_joint_offset",
            "arm_right_5_joint_offset", "arm_right_6_joint_offset",
            "arm_right_7_joint_offset",
        ]},
        **offsets,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(xacro_template.format(**values))
    print(f"\nCalibration xacro written to:\n  {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Figaroh geometric calibration for Tiago Pro right arm."
    )
    parser.add_argument("--urdf",   required=True,          help="Path to the Tiago Pro URDF")
    parser.add_argument("--data",   default=str(_DATA_DEFAULT))
    parser.add_argument("--config", default=str(_CONFIG))
    parser.add_argument("--output", default=str(_OUT_DEFAULT))
    parser.add_argument("--no-write", action="store_true",
                        help="Run calibration but do not write the xacro file")
    args = parser.parse_args()

    print(f"Loading robot from {args.urdf} ...")
    robot = _Robot(pin.buildModelFromUrdf(args.urdf))

    calib = TiagoProCalibration(robot, args.config, args.data)
    calib.load_and_check_data()
    calib.solve()

    if not args.no_write:
        write_calibration_xacro(calib, args.output)
    else:
        print("\n--no-write: xacro not modified.")


if __name__ == "__main__":
    main()
