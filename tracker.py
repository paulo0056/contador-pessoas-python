import math


class Tracker:
    def __init__(self):
        #Guarda as posições centrais do objeto
      
        self.center_points = {}
        # Mantem a contagem dos ID's
        # Toda vez que um novo id de objeto é detectad, o contador vai aumentar em 1
        self.id_count = 0


    def update(self, objects_rect):
        # Caixa dos objetos e os ID's
        objects_bbs_ids = []

        # pega o ponto do centro de um objeto
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Descobre se o objeto ja foi detectado antes
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
#                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # Se um novo objeto é detectado , atribui um novo ID a ele
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Limpa o dicionario por pontos centrais para remover IDS que não são mais usados
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Atualiza o dicionario com IDs não usados ​​removidos
        self.center_points = new_center_points.copy()
        return objects_bbs_ids