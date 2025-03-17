from tortoise import fields, models

class UserBuffer(models.Model):
    user_id = fields.CharField(max_length=255, pk=True)
    buffer_data = fields.BinaryField(null=True)
    user_context = fields.TextField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "user_buffers"