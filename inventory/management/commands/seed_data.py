"""
python manage.py seed_data
Crée groupes, utilisateurs de test, données opérationnelles et lie les SKUs ML.
"""
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User, Group
from inventory.models import Category, Product


class Command(BaseCommand):
    help = 'Seed groupes, utilisateurs et données de démonstration StockFlow Intelligence'

    def handle(self, *args, **options):
        # Groupes
        for name in ['superadmin', 'admin', 'viewer']:
            Group.objects.get_or_create(name=name)
        self.stdout.write(self.style.SUCCESS('Groupes créés'))

        # Utilisateurs
        accounts = [
            ('super1', 'super1@stockflow.ma', 'super1pass', True,  'superadmin'),
            ('admin1', 'admin1@stockflow.ma', 'admin1pass', False, 'admin'),
            ('user1',  'user1@stockflow.ma',  'user1pass',  False, 'viewer'),
        ]
        for username, email, pwd, is_super, group_name in accounts:
            if not User.objects.filter(username=username).exists():
                if is_super:
                    u = User.objects.create_superuser(username, email, pwd)
                else:
                    u = User.objects.create_user(username, email, pwd)
                u.groups.add(Group.objects.get(name=group_name))
        self.stdout.write(self.style.SUCCESS('Utilisateurs créés'))

        # Catégories
        cats = {}
        for name, desc in [
            ('Composants électroniques', 'Résistances, condensateurs, microcontrôleurs'),
            ('Matières premières',       'Métaux, polymères, liquides industriels'),
            ('Consommables',             'EPI, lubrifiants, visserie'),
            ('Équipements',              'Pièces de rechange, capteurs, actionneurs'),
        ]:
            c, _ = Category.objects.get_or_create(name=name, defaults={'description': desc})
            cats[name] = c
        self.stdout.write(self.style.SUCCESS('Catégories créées'))

        # Produits avec lien SKU ML
        products = [
            ('Résistance 10kΩ lot 1000', cats['Composants électroniques'], 12.50,  500, 50,  'COMP-001'),
            ('Microcontrôleur STM32F4',  cats['Composants électroniques'], 85.00,  40,  5,   'COMP-003'),
            ('Module ESP32-WROOM',       cats['Composants électroniques'], 45.00,  25,  5,   'COMP-009'),
            ('Acier inox 316L (kg)',     cats['Matières premières'],        180.00, 200, 30,  'MAT-001'),
            ('Aluminium 6061 (kg)',      cats['Matières premières'],        95.00,  150, 25,  'MAT-002'),
            ('Gants nitrile boîte 100',  cats['Consommables'],              35.00,  80,  20,  'CON-001'),
            ('Vis inox M6x20 lot 500',   cats['Consommables'],              28.00,  200, 30,  'CON-008'),
            ('Roulement SKF 6204',       cats['Équipements'],               120.00, 15,  3,   'EQP-001'),
            ('Vanne solénoïde 24V',      cats['Équipements'],               380.00, 8,   2,   'EQP-005'),
            ('Variateur freq 2.2kW',     cats['Équipements'],               1850.00,4,   1,   'EQP-010'),
        ]
        for name, cat, price, stock, threshold, ml_id in products:
            Product.objects.get_or_create(
                name=name,
                defaults={
                    'category':        cat,
                    'price':           price,
                    'stock':           stock,
                    'alert_threshold': threshold,
                    'ml_sku_id':       ml_id,
                    'description':     f'Produit industriel — SKU ML : {ml_id}',
                }
            )
        self.stdout.write(self.style.SUCCESS('Produits créés avec liens SKU ML'))
        self.stdout.write('')
        self.stdout.write('  super1 / super1pass  → superadmin')
        self.stdout.write('  admin1 / admin1pass  → admin')
        self.stdout.write('  user1  / user1pass   → viewer')
