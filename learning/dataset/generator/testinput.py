class STI():
    def __init__(self, sti_id, running_cpu):
        if isinstance(sti_id, int):
            sti_id = str(sti_id)
        assert isinstance(sti_id, str) is True
        assert running_cpu in ["cpu0", "cpu1"]
        self.id = sti_id
        self.cpu = running_cpu

    def __eq__(self, cmp):
        if isinstance(cmp, self.__class__) is False:
            return False
        if self.id != getattr(cmp, "id"):
            return False
        if self.cpu != getattr(cmp, "cpu"):
            return False
        return True


class CTI():
    def __init__(self, sti_a, sti_b):
        assert isinstance(sti_a, STI) is True
        assert isinstance(sti_b, STI) is True
        assert sti_a.cpu != sti_b.cpu

        self.sti_by_cpu = {}
        self.sti_by_cpu[sti_a.cpu] = sti_a
        self.sti_by_cpu[sti_b.cpu] = sti_b
        self.id = f"{sti_a.id}_{sti_b.id}"

    def __eq__(self, cmp):
        if isinstance(cmp, self.__class__) is False:
            return False
        sti_by_cpu = getattr(cmp, "sti_by_cpu")
        for cpu in self.sti_by_cpu:
            if self.sti_by_cpu[cpu] != sti_by_cpu[cpu]:
                return False
        return True


class SCHEDULE():
    def __init__(self, init_cpu, cpu0_switch_point, cpu1_switch_point):
        assert init_cpu in ["cpu0", "cpu1"]
        assert isinstance(cpu0_switch_point, int) is True
        assert isinstance(cpu1_switch_point, int) is True
        self.init_cpu = init_cpu
        self.cpu0_switch_point = cpu0_switch_point
        self.cpu1_switch_point = cpu1_switch_point
        if init_cpu == "cpu0":
            self.id = f"0_{hex(cpu0_switch_point)[2:]}_{hex(cpu1_switch_point)[2:]}"
        else:
            self.id = f"1_{hex(cpu0_switch_point)[2:]}_{hex(cpu1_switch_point)[2:]}"

    def __eq__(self, cmp):
        if isinstance(cmp, self.__class__) is False:
            return False
        init_cpu = getattr(cmp, "init_cpu")
        if init_cpu != self.init_cpu:
            return False
        if self.cpu0_switch_point != getattr(cmp, "cpu0_switch_point"):
            return False
        if self.cpu1_switch_point != getattr(cmp, "cpu1_switch_point"):
            return False
        return True


class CT():
    def __init__(self, cti, schedule, coverage_by_cpu=None):
        assert isinstance(cti, CTI) is True
        assert isinstance(schedule, SCHEDULE) is True
        self.cti = cti
        self.schedule = schedule
        if coverage_by_cpu is not None:
            assert "cpu0" in coverage_by_cpu
            assert "cpu1" in coverage_by_cpu
        self.coverage_by_cpu = coverage_by_cpu
        self.id = f"{cti.id}_{schedule.id}"

    def __eq__(self, cmp):
        if isinstance(cmp, self.__class__) is False:
            return False
        if self.cti != getattr(cmp, "cti"):
            return False
        schedule = getattr(cmp, "schedule")
        if schedule != self.schedule:
            return False
        return True
