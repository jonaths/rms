
import math


class RmsAlg:

    def __init__(self, rthres, influence, risk_default, sim_func_name='manhattan', risk_func_name='inverse'):
        self.rthres = rthres
        self.influence = influence
        self.risk_default = risk_default
        self.sim_func_name = sim_func_name
        self.risk_func_name = risk_func_name
        self.k = {}
        self.v = {}
        self.kr = {}

    def add_to_k(self, index):
        """
        Agrega k a K si no existe
        :param index:
        :return:
        """
        if index not in self.k:
            self.k[index] = {}

    def add_to_v(self, index, features):
        """
        Agrega v a V si no existe
        :param index:
        :return:
        """
        if index not in self.v:
            self.v[index] = features

    def calc_difference(self, s, sprime):
        """
        Calcula la similaridad entre dos estados s
        :param s: un vector [a, b, c...]
        :param sprime: otro vector de la misma dimension que s
        :return: escalar con la similaridad
        """

        def manhattan_distance(start, end):
            # print(start, end)
            return sum(abs(e1 - s1) for s1, e1 in zip(start, end))

        if self.sim_func_name == 'manhattan':
            difference = manhattan_distance(s, sprime)
        else:
            raise 'Invalid sim_func'
        return difference if difference <= self.influence else float('inf')

    def calc_risk(self, s, sprime):
        """
        Calcula el riesgo en funcion de la medida de similaridad predefinida
        :param s: un vector [a, b, c... ]
        :param sprime: otro vector de la misma dimension que s
        :return: un escalar con el riesgo.
        """

        def inverse(diff):
            return 1. / (diff + 1)

        difference = self.calc_difference(s, sprime)

        if self.risk_func_name == 'inverse':
            risk = inverse(difference)
        else:
            raise Exception('Invalid risk func')

        # print(difference, risk)
        return risk

    def update(self, s, r, sprime, sprime_features=None):
        """
        Actualiza la tabla de riesgos considerando una nueva transicion de s a
        sprime.
        :param s: el estado actual (debe existir ya en v)
        :param r: la recompensa de la transicion
        :param sprime: el estado siguiente
        :param sprime_features: los features del estado siguiente (si no esta ya en v)
        :return:
        """
        if s not in self.v:
            raise Exception("Nonexistent state: {}. v: {}".format(s, self.v))
        # verifica que el estado este en la lista de conocidos
        self.add_to_v(sprime, sprime_features)
        # verifica si es necesario agregarlo a la lisa de riesgosos
        if r < self.rthres:
            self.add_to_k(sprime)
            self.kr[sprime] = r
        for ki in self.k:
            for vi in self.v:
                # calcula el riesgo de todas las combinaciones
                risk = self.calc_risk(self.v[ki], self.v[vi])
                # print("risk:", ki, vi, risk)
                # actualiza el valor actual de k
                self.k[ki][vi] = risk

    def get_risk(self, state):
        """
        Calcula todos los riesgos que influyen en un estado considerando la recompensa
        en el centro
        :param state: el estado del que se quiere saber el riesgo
        :return:
        """
        total_risk = self.risk_default
        for ki in self.k:
            if state in self.k[ki]:
                total_risk += self.k[ki][state] * self.kr[ki]
        return total_risk

    def get_risk_dict(self):
        risk_dict = {}
        for index, val in self.v.items():
            risk_dict[index] = self.get_risk(index)
        return risk_dict

    def get_risk_dict_no_zeros(self):
        risk_dict = {}
        for index, val in self.v.items():
            risk = self.get_risk(index)
            if risk != 0:
                risk_dict[index] = risk
        return risk_dict
