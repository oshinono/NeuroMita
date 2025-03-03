using UnityEngine;
using System.Collections;
using MelonLoader;

namespace MitaAI
{

    [RegisterTypeInIl2Cpp]
    public class WardrobeFix : MonoBehaviour
    {
        private float moveDuration = 3f; // Время перемещения
        private float moveDistance = 4f; // Дистанция перемещения

        private void OnTriggerEnter(Collider other)
        {
            try
            {
                MelonLogger.Msg($"Deect {other.gameObject.name}");
                // Определяем направление по Z
                if (other || !other.gameObject.name.Contains("Mita")) return;

                MelonLogger.Msg("Mita");
                float zDirection = (other.transform.position.z > transform.position.z) ? 1 : -1;

                // Запускаем последовательность перемещений
                MelonCoroutines.Start(MovementSequence(zDirection, other.gameObject));
            }
            catch (Exception e)
            {

                MelonLogger.Error($"Deect error {e}");
            }
           
        }

        private IEnumerator MovementSequence(float direction, GameObject targetObject)
        {
            // Первый этап: движение к центру
            yield return MelonCoroutines.Start(MoveObject(targetObject.transform,direction * moveDistance,-direction,moveDuration));

            // Второй этап: движение за край
            yield return MelonCoroutines.Start(MoveObject(targetObject.transform,-direction * moveDistance,direction,moveDuration));
        }

        private IEnumerator MoveObject(Transform obj, float distance, float direction, float duration)
        {
            Vector3 startPos = obj.position;
            Vector3 endPos = startPos + new Vector3(0, 0, distance);

            float elapsed = 0f;

            while (elapsed < duration)
            {
                obj.position = Vector3.Lerp(startPos, endPos, elapsed / duration);
                elapsed += Time.deltaTime;
                yield return null;
            }

            obj.position = endPos;
        }
    }
}